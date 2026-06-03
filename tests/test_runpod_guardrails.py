import json
import sys
from pathlib import Path

import pytest

# Ensure scripts/ namespace imports work when tests run from repo root or elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import runpod_launch
from scripts import runpod_supervisor as supervisor
from scripts import train_watchdog


def _manifest(**overrides):
    data = dict(
        job_id="20260603T000000Z-d12-grkan-g8",
        model_tag="d12-grkan-g8-corrected",
        gpu="NVIDIA H100 80GB HBM3",
        cloud_type="SECURE",
        volume_id="vol_123",
        repo_url="https://github.com/ACS-USP/nanokan",
        repo_ref="abcdef123456",
        rational_kat_cu_ref="41a20b5",
        command="python scripts/runpod_launch.py train ...",
        max_runtime_minutes=30,
        max_cost_usd=1.0,
        expected_artifacts=["run_manifest.json"],
        local_dest="checkpoints/nanochat",
    )
    data.update(overrides)
    return supervisor.RunManifest(**data)


@pytest.mark.parametrize(
    ("line", "kind", "reason"),
    [
        ("step 00017/02520 | loss: nan | lrm: 0.42", "nonfinite_loss", "loss_nan"),
        ("Step 00010 | Validation bpb: NaN", "nonfinite_validation", "validation_nan"),
        ("RUNPOD_GUARD_FAIL reason=nonfinite_loss step=17 micro_step=3 model_tag=d12", "guard_fail", "nonfinite_loss"),
        ("Traceback (most recent call last):", "traceback", "traceback"),
        ("RuntimeError: CUDA error: device-side assert triggered", "cuda_error", "cuda_error"),
    ],
)
def test_log_parser_detects_stop_events(line, kind, reason):
    event = supervisor.parse_log_line(line)
    assert event is not None
    assert event.kind == kind
    assert event.reason == reason
    assert event.should_stop


def test_validate_train_request_blocks_unsafe_science_runs():
    with pytest.raises(ValueError, match="non-smoke training requires --volume-id"):
        supervisor.validate_train_request(smoke=False, volume_id=None, rational_kat_cu_ref="41a20b5")
    with pytest.raises(ValueError, match="pinned"):
        supervisor.validate_train_request(smoke=True, volume_id=None, rational_kat_cu_ref="main")
    supervisor.validate_train_request(smoke=True, volume_id=None, rational_kat_cu_ref="41a20b5")
    supervisor.validate_train_request(smoke=False, volume_id="vol_123", rational_kat_cu_ref="41a20b5")


def test_manifest_roundtrip_and_state_transitions(tmp_path):
    manifest = _manifest()
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = supervisor.RunManifest.load(path)
    assert loaded.job_id == manifest.job_id
    loaded.set_pod("pod_123", gpu="NVIDIA H100 PCIe")
    loaded.transition("running", note="ssh ready")
    loaded.transition("stopping", note="guard failure")
    loaded.transition("stopped", note="compute stopped")
    loaded.save(path)
    persisted = json.loads(path.read_text())
    assert persisted["pod_id"] == "pod_123"
    assert persisted["state"] == "stopped"
    with pytest.raises(ValueError, match="state regression"):
        loaded.transition("running")


def test_train_startup_is_pinned_failfast_and_job_scoped(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setattr(runpod_launch, "_read_ssh_pubkey", lambda: "ssh-ed25519 AAAATEST")
    monkeypatch.setattr(runpod_launch, "_local_source_hotfix", lambda: [])

    startup = runpod_launch._make_train_startup(
        depth=12,
        ffn_type="grkan",
        nanochat_base="/runpod-volume/nanochat/jobs/job123",
        run_name="d12-grkan-g16-corrected",
        model_tag_override="d12-grkan-g16-corrected",
        grkan_groups=16,
        smoke=False,
        save_every_override=500,
        num_iterations=2520,
        repo_ref="abcdef123456",
        rational_kat_cu_ref="41a20b5",
        job_id="job123",
        max_runtime_minutes=30,
    )
    script = "\n".join(startup)
    assert "rational_kat_cu.git@41a20b5" in script
    assert "git checkout abcdef123456" in script
    assert "/runpod-volume/nanochat/jobs/job123" in script
    assert "scripts/train_watchdog.py" in script
    assert "--fail-fast" in script
    assert "--debug-finite-steps=20" in script
    assert "--grkan-groups=16" in script
    assert "runpodctl stop pod $RUNPOD_POD_ID" in script


def test_watchdog_writes_failure_bundle_on_nan(tmp_path):
    log_path = tmp_path / "train.log"
    artifact_dir = tmp_path / "artifacts"
    cmd = [sys.executable, "-c", "print('step 00000/00020 | loss: nan', flush=True)"]
    rc = train_watchdog.run_watchdog(
        train_cmd=cmd,
        log_path=log_path,
        artifact_dir=artifact_dir,
        model_tag="d12-grkan-g8-corrected",
        max_runtime_minutes=None,
        heartbeat_timeout_minutes=1,
        poll_seconds=0.01,
        stop_pod_on_failure=False,
    )
    assert rc == 42
    failure_json = artifact_dir / "failure" / "d12-grkan-g8-corrected" / "failure.json"
    failed_sentinel = artifact_dir / "FAILED_d12-grkan-g8-corrected"
    assert failure_json.exists()
    assert failed_sentinel.exists()
    payload = json.loads(failure_json.read_text())
    assert payload["reason"] == "loss_nan"
