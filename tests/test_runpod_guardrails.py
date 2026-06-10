import base64
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure scripts/ namespace imports work when tests run from repo root or elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import runpod_launch
from scripts import runpod_supervisor as supervisor
from scripts import train_watchdog


def _decoded_startup_payload(script: str, path_fragment: str) -> str:
    pattern = rf"echo ([A-Za-z0-9+/=]+) \| base64 -d > {re.escape(path_fragment)}"
    match = re.search(pattern, script)
    assert match is not None
    return base64.b64decode(match.group(1)).decode()


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
    install_rat = _decoded_startup_payload(script, "/tmp/install_rat.sh")
    assert "rational_kat_cu.git@41a20b5" in install_rat
    assert "git checkout abcdef123456" in script
    assert "/runpod-volume/nanochat/jobs/job123" in script
    assert "scripts/train_watchdog.py" in script
    assert "--fail-fast" in script
    assert "--debug-finite-steps=20" in script
    assert "--grkan-groups=16" in script
    assert "runpodctl stop pod $RUNPOD_POD_ID" in script


def test_train_autosupervises_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setattr(runpod_launch, "get_api_key", lambda: "rp_test")
    monkeypatch.setattr(runpod_launch, "_read_ssh_pubkey", lambda: "ssh-ed25519 AAAATEST")
    monkeypatch.setattr(runpod_launch, "_local_source_hotfix", lambda: [])
    monkeypatch.setattr(runpod_launch, "_local_git_ref", lambda: "abcdef123456")
    monkeypatch.setattr(runpod_launch, "_dirty_patch_sha256", lambda: None)
    monkeypatch.setattr(
        runpod_launch,
        "find_and_launch_pod",
        lambda **_kwargs: ("NVIDIA H100 80GB HBM3", {"id": "pod_123", "costPerHr": 3.29}),
    )

    captured = {}

    def fake_supervise(args):
        captured["manifest"] = args.manifest
        captured["terminate_on_done"] = args.terminate_on_done

    monkeypatch.setattr(runpod_launch, "cmd_supervise", fake_supervise)

    args = runpod_launch.build_parser().parse_args(
        [
            "train",
            "--ffn-type",
            "grkan",
            "--grkan-groups",
            "16",
            "--num-iterations",
            "500",
            "--volume-id",
            "vol_123",
            "--gate-approved",
            "--manifest-dir",
            str(tmp_path),
        ]
    )
    runpod_launch.cmd_train(args)

    assert captured["terminate_on_done"] is True
    manifest_path = Path(captured["manifest"])
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["pod_id"] == "pod_123"
    assert payload["model_tag"] == "d12-grkan-g16"


def test_train_detach_skips_autosupervise(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setattr(runpod_launch, "get_api_key", lambda: "rp_test")
    monkeypatch.setattr(runpod_launch, "_read_ssh_pubkey", lambda: "ssh-ed25519 AAAATEST")
    monkeypatch.setattr(runpod_launch, "_local_source_hotfix", lambda: [])
    monkeypatch.setattr(runpod_launch, "_local_git_ref", lambda: "abcdef123456")
    monkeypatch.setattr(runpod_launch, "_dirty_patch_sha256", lambda: None)
    monkeypatch.setattr(
        runpod_launch,
        "find_and_launch_pod",
        lambda **_kwargs: ("NVIDIA H100 80GB HBM3", {"id": "pod_123", "costPerHr": 3.29}),
    )
    monkeypatch.setattr(runpod_launch, "cmd_supervise", lambda _args: pytest.fail("unexpected supervise"))

    args = runpod_launch.build_parser().parse_args(
        [
            "train",
            "--ffn-type",
            "grkan",
            "--grkan-groups",
            "16",
            "--num-iterations",
            "500",
            "--volume-id",
            "vol_123",
            "--gate-approved",
            "--manifest-dir",
            str(tmp_path),
            "--detach",
        ]
    )
    runpod_launch.cmd_train(args)

    manifests = list(tmp_path.glob("*.json"))
    assert len(manifests) == 1


def test_watch_and_supervise_terminate_by_default():
    parser = runpod_launch.build_parser()
    watch_args = parser.parse_args(["watch", "pod_123"])
    supervise_args = parser.parse_args(["supervise", "--manifest", "manifest.json"])
    keep_watch_args = parser.parse_args(["watch", "pod_123", "--keep-pod-on-done"])
    keep_supervise_args = parser.parse_args(["supervise", "--manifest", "manifest.json", "--keep-pod-on-done"])

    assert watch_args.terminate_on_done is True
    assert supervise_args.terminate_on_done is True
    assert keep_watch_args.terminate_on_done is False
    assert keep_supervise_args.terminate_on_done is False


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


class _FakeRunpod:
    def __init__(self):
        self.stopped = []
        self.terminated = []

    def get_pod(self, pod_id):
        return {
            "desiredStatus": "RUNNING",
            "runtime": {"ports": [{"privatePort": 22, "ip": "1.2.3.4", "publicPort": 2222}]},
        }

    def stop_pod(self, pod_id):
        self.stopped.append(pod_id)

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


def _watch_args(**overrides):
    class A:
        pass

    a = A()
    a.pod_id = "pod_test"
    a.dest = "/tmp/nanokan-watch-test"
    a.interval = 1
    a.volume_backed = False
    a.ssh_failure_limit = 6
    a.terminate_on_done = True
    a.manifest_path = None
    a.setup_deadline_minutes = 1
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _install_watch_mocks(monkeypatch, fake_runpod, *, log_stdout, status_stdout):
    monkeypatch.setattr(runpod_launch, "get_api_key", lambda: "rp_test")
    monkeypatch.setattr(runpod_launch, "runpod", fake_runpod)

    def fake_run(cmd, capture_output=False, text=False, **kwargs):
        joined = " ".join(cmd)
        if "echo ok" in joined:                        # sshd readiness probe
            return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")
        if "_train.log" in joined:                     # remote log streaming
            return subprocess.CompletedProcess(cmd, 0, stdout=log_stdout, stderr="")
        if "DONE_" in joined or "FAILED_" in joined:    # sentinel status check
            return subprocess.CompletedProcess(cmd, 0, stdout=status_stdout, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")  # rsync etc.

    monkeypatch.setattr(runpod_launch.subprocess, "run", fake_run)
    # First reading arms the deadline at t=0; the next reading is far past it.
    clock = itertools.chain([0.0], itertools.repeat(1e9))
    monkeypatch.setattr("time.monotonic", lambda: next(clock))
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    monkeypatch.setattr(sys.stdout, "reconfigure", lambda **k: None, raising=False)


def test_watch_stops_pod_when_setup_stalls(monkeypatch):
    # No training-log output: the train script never started (e.g. uv sync stall).
    fake = _FakeRunpod()
    _install_watch_mocks(monkeypatch, fake, log_stdout="", status_stdout="running\n")

    with pytest.raises(SystemExit):
        runpod_launch.cmd_watch(_watch_args(pod_id="pod_stuck"))

    assert fake.stopped == ["pod_stuck"]
    assert fake.terminated == []


def test_watch_does_not_stop_after_training_starts(monkeypatch):
    # Training log has step output → training started; even with the clock past the
    # deadline the setup guard must not fire.  The DONE sentinel ends the watch cleanly.
    fake = _FakeRunpod()
    monkeypatch.setattr(runpod_launch, "cmd_download", lambda _a: None)
    _install_watch_mocks(
        monkeypatch, fake,
        log_stdout="step 00000/00020 | loss: 10.40\n",
        status_stdout="done\n",
    )

    runpod_launch.cmd_watch(_watch_args(pod_id="pod_ok"))

    assert fake.stopped == []


def test_setup_deadline_flag_is_wired():
    parser = runpod_launch.build_parser()
    for argv in (["watch", "pod_1"], ["supervise", "--manifest", "m.json"], ["train", "--ffn-type", "grkan"]):
        ns = parser.parse_args(argv)
        assert hasattr(ns, "setup_deadline_minutes")
        assert ns.setup_deadline_minutes is None  # default → cmd_watch falls back to SETUP_DEADLINE_MINUTES
    override = parser.parse_args(["watch", "pod_1", "--setup-deadline-minutes", "10"])
    assert override.setup_deadline_minutes == 10.0
