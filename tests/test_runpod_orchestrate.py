import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import runpod_orchestrate as orchestrate


def test_parse_models_infers_grkan_groups_and_mlp():
    models = orchestrate._parse_models("d12-grkan-g4,d12-grkan-g16,d12-mlp")

    assert [(m.tag, m.ffn_type, m.depth, m.grkan_groups) for m in models] == [
        ("d12-grkan-g4", "grkan", 12, 4),
        ("d12-grkan-g16", "grkan", 12, 16),
        ("d12-mlp", "mlp", 12, None),
    ]


def test_plan_builds_sequential_smoke_then_pilot_workflow(tmp_path, capsys):
    rc = orchestrate.main(
        [
            "plan",
            "--models",
            "d12-grkan-g4,d12-grkan-g16",
            "--repo-ref",
            "abcdef123456",
            "--rational-kat-cu-ref",
            "41a20b5",
            "--volume-id",
            "vol_123",
            "--until",
            "pilot",
            "--workflow-id",
            "wf-test",
            "--workflow-dir",
            str(tmp_path / "workflow"),
            "--manifest-dir",
            str(tmp_path / "runpod_manifests"),
            "--local-dest",
            str(tmp_path / "nanochat"),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN:" in out
    manifest_path = tmp_path / "workflow" / "wf-test.json"
    payload = json.loads(manifest_path.read_text())
    assert payload["state"] == "planned"
    assert [(r["model_tag"], r["phase"]) for r in payload["runs"]] == [
        ("d12-grkan-g4", "smoke"),
        ("d12-grkan-g16", "smoke"),
        ("d12-grkan-g4", "pilot"),
        ("d12-grkan-g16", "pilot"),
    ]


def test_plan_blocks_unpinned_refs(tmp_path):
    rc = orchestrate.main(
        [
            "plan",
            "--models",
            "d12-grkan-g16",
            "--repo-ref",
            "main",
            "--volume-id",
            "vol_123",
            "--workflow-dir",
            str(tmp_path),
        ]
    )

    assert rc == 2


def test_verify_artifacts_rejects_missing_rational_kernel_confirmation(tmp_path):
    model = orchestrate.ModelSpec("d12-grkan-g16", "grkan", 12, 16)
    phase = orchestrate.PhaseSpec(
        name="smoke",
        kind="smoke",
        num_iterations=20,
        save_every=10,
        max_runtime_minutes=30,
        requires_gate=False,
        requires_volume=False,
    )
    artifact_dir = tmp_path / "artifacts"
    ckpt_dir = artifact_dir / model.tag
    tok_dir = artifact_dir / "tokenizer"
    ckpt_dir.mkdir(parents=True)
    tok_dir.mkdir(parents=True)
    (artifact_dir / "run_manifest.json").write_text("{}")
    (artifact_dir / f"{model.tag}_train.log").write_text("rational_kat_cu not installed\n")
    (tok_dir / "tokenizer.pkl").write_text("tok")
    (tok_dir / "token_bytes.pt").write_text("tok")
    (ckpt_dir / "model_000020.pt").write_text("model")
    (ckpt_dir / "optim_000020_rank0.pt").write_text("optim")
    (ckpt_dir / "meta_000020.json").write_text(
        json.dumps({"step": 20, "val_bpb": 1.0, "dataloader_state_dict": {"epoch": 1}})
    )
    launcher_manifest = tmp_path / "phase.json"
    launcher_manifest.write_text(
        json.dumps(
            {
                "state": "terminated",
                "expected_artifacts": [
                    "run_manifest.json",
                    f"{model.tag}_train.log",
                    "tokenizer/tokenizer.pkl",
                    "tokenizer/token_bytes.pt",
                    f"base_checkpoints/{model.tag}/model_000020.pt",
                    f"base_checkpoints/{model.tag}/meta_000020.json",
                ],
            }
        )
    )
    phase_run = orchestrate.PhaseRun(
        model_tag=model.tag,
        phase="smoke",
        job_id="job",
        manifest_path=str(launcher_manifest),
        artifact_dir=str(artifact_dir),
    )

    with pytest.raises(orchestrate.WorkflowError, match="rational_kat_cu"):
        orchestrate._verify_artifacts(phase_run, model, phase)


def test_resume_existing_phase_supervises_instead_of_relaunching(monkeypatch, tmp_path):
    model = orchestrate.ModelSpec("d12-grkan-g16", "grkan", 12, 16)
    phase = orchestrate.PhaseSpec(
        name="pilot",
        kind="pilot",
        num_iterations=500,
        save_every=500,
        max_runtime_minutes=90,
        requires_gate=True,
        requires_volume=True,
        requires_previous="smoke",
    )
    launcher_manifest = tmp_path / "phase.json"
    launcher_manifest.write_text(json.dumps({"state": "running", "pod_id": "pod_123"}))
    phase_run = orchestrate.PhaseRun(
        model_tag=model.tag,
        phase=phase.name,
        job_id="wf-pilot-d12-grkan-g16",
        manifest_path=str(launcher_manifest),
        artifact_dir=str(tmp_path / "artifacts"),
    )
    captured = {}

    def fake_run(cmd, *, execute, capture=False):
        captured["cmd"] = cmd
        return orchestrate.subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(orchestrate, "_run", fake_run)

    result = orchestrate._run_or_resume_phase(
        phase_run=phase_run,
        model=model,
        phase=phase,
        phase_manifest_dir=tmp_path,
        args=Namespace(),
        execute=True,
    )

    assert result.returncode == 0
    assert "supervise" in captured["cmd"]
    assert "train" not in captured["cmd"]


def test_resume_terminal_good_phase_verifies_without_runpod_call(monkeypatch, tmp_path):
    model = orchestrate.ModelSpec("d12-grkan-g16", "grkan", 12, 16)
    phase = orchestrate.PhaseSpec("smoke", "smoke", 20, 10, 30, False, False)
    launcher_manifest = tmp_path / "phase.json"
    launcher_manifest.write_text(json.dumps({"state": "terminated"}))
    phase_run = orchestrate.PhaseRun(
        model_tag=model.tag,
        phase=phase.name,
        job_id="wf-smoke-d12-grkan-g16",
        manifest_path=str(launcher_manifest),
        artifact_dir=str(tmp_path / "artifacts"),
    )
    monkeypatch.setattr(orchestrate, "_run", lambda *_args, **_kwargs: pytest.fail("unexpected RunPod command"))

    result = orchestrate._run_or_resume_phase(
        phase_run=phase_run,
        model=model,
        phase=phase,
        phase_manifest_dir=tmp_path,
        args=Namespace(),
        execute=True,
    )

    assert result.returncode == 0
    assert result.args == ["verify-existing", str(launcher_manifest)]


def test_preflight_allows_only_manifest_recorded_resume_pods(monkeypatch):
    ls_stdout = (
        "Name                                ID                   GPU                    $/hr  Status\n"
        "------------------------------------------------------------------------------------------\n"
        "nanochat-safe                       pod_allowed          NVIDIA H100 80GB HBM3   $3.29  RUNNING\n"
    )

    monkeypatch.setattr(
        orchestrate,
        "_run",
        lambda *_args, **_kwargs: orchestrate.subprocess.CompletedProcess(["ls"], 0, stdout=ls_stdout, stderr=""),
    )
    args = Namespace(allow_existing_pods=False)

    orchestrate._assert_no_pods(args, execute=True, context="preflight", allowed_pod_ids={"pod_allowed"})

    with pytest.raises(orchestrate.WorkflowError, match="stale RunPod pods"):
        orchestrate._assert_no_pods(args, execute=True, context="preflight", allowed_pod_ids={"other_pod"})


def test_preflight_rejects_diverging_local_dest(tmp_path):
    args = Namespace(
        repo_ref="abcdef123456",
        rational_kat_cu_ref="41a20b5",
        volume_id=None,
        local_dest=str(tmp_path / "elsewhere"),
        allow_existing_pods=False,
    )
    phases = [orchestrate.PhaseSpec("smoke", "smoke", 20, 10, 30, False, False)]

    with pytest.raises(orchestrate.WorkflowError, match="local-dest"):
        orchestrate._preflight(args, phases, execute=True)


def test_preflight_accepts_launcher_local_dest(monkeypatch):
    # Default local-dest matches the launcher download root, so the dest guard
    # passes and preflight proceeds to the credential check.
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    args = Namespace(
        repo_ref="abcdef123456",
        rational_kat_cu_ref="41a20b5",
        volume_id=None,
        local_dest=str(orchestrate.DEFAULT_LOCAL_DEST),
        allow_existing_pods=False,
    )
    phases = [orchestrate.PhaseSpec("smoke", "smoke", 20, 10, 30, False, False)]

    with pytest.raises(orchestrate.WorkflowError, match="RUNPOD_API_KEY"):
        orchestrate._preflight(args, phases, execute=True)


def _ls_with_pod(pod_id: str) -> str:
    return (
        "Name                                ID                   GPU                    $/hr  Status\n"
        "------------------------------------------------------------------------------------------\n"
        f"nanochat-d12-grkan-g4               {pod_id}          NVIDIA H100 80GB HBM3   $3.29  RUNNING\n"
    )


def test_assert_no_pods_grace_waits_for_termination(monkeypatch):
    # First poll still lists the just-terminated pod; the second poll is clear.
    outputs = [_ls_with_pod("pod_dying"), "No pods.\n"]
    sleeps = []

    def fake_run(cmd, *, execute, capture=False):
        return orchestrate.subprocess.CompletedProcess(cmd, 0, stdout=outputs.pop(0), stderr="")

    monkeypatch.setattr(orchestrate, "_run", fake_run)
    monkeypatch.setattr(orchestrate.time, "sleep", lambda *_a, **_k: sleeps.append(1))
    args = Namespace(allow_existing_pods=False)

    orchestrate._assert_no_pods(args, execute=True, context="post", settle_attempts=3, settle_seconds=0)

    assert outputs == []  # exactly two polls consumed (a third would IndexError)
    assert sleeps == [1]  # one grace wait between the two polls


def test_assert_no_pods_grace_raises_when_pod_persists(monkeypatch):
    monkeypatch.setattr(
        orchestrate,
        "_run",
        lambda *_a, **_k: orchestrate.subprocess.CompletedProcess(["ls"], 0, stdout=_ls_with_pod("pod_stuck"), stderr=""),
    )
    monkeypatch.setattr(orchestrate.time, "sleep", lambda *_a, **_k: None)
    args = Namespace(allow_existing_pods=False)

    with pytest.raises(orchestrate.WorkflowError, match="stale RunPod pods"):
        orchestrate._assert_no_pods(args, execute=True, context="post", settle_attempts=2, settle_seconds=0)
