"""Top-level RunPod workflow orchestrator for safe nanochat campaigns.

This script owns the sequence around ``runpod_launch.py``:

1. preflight: no stale pods, pinned refs, required volume;
2. smoke: exact 20-step training smoke;
3. pilot: first-save pilot with artifact retrieval;
4. full: only after smoke and pilot pass local finite checks;
5. resume: recover from the manifest instead of relaunching blind.

The low-level launcher remains responsible for pod creation, the in-pod
watchdog, local supervision, downloads, and termination.  This layer turns the
RunPod guide's sequencing rules into mandatory state transitions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.runpod_supervisor import first_guard_event, is_pinned_git_ref, make_job_id, utc_now_iso
except ImportError:  # pragma: no cover - direct script execution path
    from runpod_supervisor import first_guard_event, is_pinned_git_ref, make_job_id, utc_now_iso  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "runpod_launch.py"
DEFAULT_LOCAL_DEST = REPO_ROOT / "checkpoints" / "nanochat"
DEFAULT_WORKFLOW_DIR = DEFAULT_LOCAL_DEST / "orchestrator"
TERMINAL_OK_STATES = {"terminated"}
TERMINAL_BAD_STATES = {"failed", "stopping", "stopped"}


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    ffn_type: str
    depth: int
    grkan_groups: int | None = None


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    kind: str
    num_iterations: int | None
    save_every: int | None
    max_runtime_minutes: float | None
    requires_gate: bool
    requires_volume: bool
    requires_previous: str | None = None


@dataclass
class PhaseRun:
    model_tag: str
    phase: str
    job_id: str
    manifest_path: str
    artifact_dir: str
    state: str = "planned"
    started_at: str | None = None
    completed_at: str | None = None
    launcher_returncode: int | None = None
    error: str | None = None


@dataclass
class WorkflowManifest:
    workflow_id: str
    command: str
    repo_ref: str
    rational_kat_cu_ref: str
    volume_id: str | None
    secure: bool
    model_specs: list[dict[str, Any]]
    phases: list[dict[str, Any]]
    runs: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    state: str = "planned"
    manifest_dir: str | None = None
    local_dest: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def event(self, kind: str, **fields: Any) -> None:
        payload = {"time": utc_now_iso(), "kind": kind}
        payload.update(fields)
        self.events.append(payload)
        self.updated_at = payload["time"]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "WorkflowManifest":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class WorkflowError(RuntimeError):
    pass


def _repo_ref() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=12", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _launcher_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Local macOS machines in this project can have unwritable default uv caches.
    env.setdefault("UV_CACHE_DIR", "/private/tmp/projectlm01-uv-cache")
    return env


def _run(cmd: list[str], *, execute: bool, capture: bool = False) -> subprocess.CompletedProcess[str]:
    printable = " ".join(cmd)
    if not execute:
        print(f"DRY-RUN: {printable}")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    print(f"RUN: {printable}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, env=_launcher_env(), text=True, capture_output=capture)


def _launcher_cmd(args: list[str]) -> list[str]:
    return [sys.executable, str(LAUNCHER), *args]


def _parse_models(raw: str) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    for part in raw.split(","):
        tag = part.strip()
        if not tag:
            continue
        if "-grkan" in tag:
            depth_text = tag.split("-", 1)[0]
            if not depth_text.startswith("d") or not depth_text[1:].isdigit():
                raise WorkflowError(f"cannot infer depth from model tag {tag!r}")
            depth = int(depth_text[1:])
            groups = 8
            for segment in tag.split("-"):
                if segment.startswith("g") and segment[1:].isdigit():
                    groups = int(segment[1:])
            models.append(ModelSpec(tag=tag, ffn_type="grkan", depth=depth, grkan_groups=groups))
        elif tag.endswith("-mlp") or "-mlp" in tag:
            depth_text = tag.split("-", 1)[0]
            if not depth_text.startswith("d") or not depth_text[1:].isdigit():
                raise WorkflowError(f"cannot infer depth from model tag {tag!r}")
            models.append(ModelSpec(tag=tag, ffn_type="mlp", depth=int(depth_text[1:])))
        else:
            raise WorkflowError(f"cannot infer model type from tag {tag!r}")
    if not models:
        raise WorkflowError("at least one model tag is required")
    return models


def _phase_specs(args: argparse.Namespace) -> list[PhaseSpec]:
    all_phases = [
        PhaseSpec(
            name="smoke",
            kind="smoke",
            num_iterations=20,
            save_every=10,
            max_runtime_minutes=args.smoke_max_runtime_minutes,
            requires_gate=False,
            requires_volume=False,
        ),
        PhaseSpec(
            name="pilot",
            kind="pilot",
            num_iterations=args.pilot_steps,
            save_every=args.pilot_steps,
            max_runtime_minutes=args.pilot_max_runtime_minutes,
            requires_gate=True,
            requires_volume=True,
            requires_previous="smoke",
        ),
        PhaseSpec(
            name="full",
            kind="full",
            num_iterations=args.full_steps,
            save_every=args.save_every,
            max_runtime_minutes=args.full_max_runtime_minutes,
            requires_gate=True,
            requires_volume=True,
            requires_previous="pilot",
        ),
    ]
    until = args.until
    names = [phase.name for phase in all_phases]
    return all_phases[: names.index(until) + 1]


def _workflow_id(models: Iterable[ModelSpec], until: str) -> str:
    model_text = "_".join(m.tag for m in models)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return make_job_id(f"{model_text}-{until}", now=datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc))


def _phase_job_id(workflow_id: str, model: ModelSpec, phase: PhaseSpec) -> str:
    return f"{workflow_id}-{phase.name}-{model.tag}"


def _phase_manifest_path(manifest_dir: Path, job_id: str) -> Path:
    return manifest_dir / f"{job_id}.json"


def _artifact_dir(local_dest: Path, job_id: str) -> Path:
    return local_dest / job_id


def _assert_pinned(label: str, value: str) -> None:
    if not is_pinned_git_ref(value):
        raise WorkflowError(f"{label} must be a pinned 7-40 hex commit, got {value!r}")


def _preflight(
    args: argparse.Namespace,
    phases: list[PhaseSpec],
    *,
    execute: bool,
    allowed_pod_ids: set[str] | None = None,
) -> None:
    _assert_pinned("repo_ref", args.repo_ref)
    _assert_pinned("rational_kat_cu_ref", args.rational_kat_cu_ref)
    if any(phase.requires_volume for phase in phases) and not args.volume_id:
        raise WorkflowError("pilot/full phases require --volume-id")

    if execute:
        if not os.environ.get("RUNPOD_API_KEY"):
            raise WorkflowError("RUNPOD_API_KEY must be exported before running the orchestrator")
        if not os.environ.get("GITHUB_TOKEN"):
            raise WorkflowError("GITHUB_TOKEN must be exported before running the orchestrator")

    _assert_no_pods(args, execute=execute, context="preflight", allowed_pod_ids=allowed_pod_ids)

    if args.volume_id and execute:
        volume_result = _run(_launcher_cmd(["volume", "ls"]), execute=True, capture=True)
        print(volume_result.stdout, end="")
        if volume_result.returncode != 0:
            raise WorkflowError("volume ls failed")
        if args.volume_id not in volume_result.stdout:
            raise WorkflowError(f"volume {args.volume_id!r} not found in RunPod volume list")


def _listed_pod_ids(ls_stdout: str) -> set[str]:
    pod_ids: set[str] = set()
    for line in ls_stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Name ") or stripped.startswith("-") or stripped == "No pods.":
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            pod_ids.add(parts[1])
    return pod_ids


def _assert_no_pods(
    args: argparse.Namespace,
    *,
    execute: bool,
    context: str,
    allowed_pod_ids: set[str] | None = None,
) -> None:
    ls_result = _run(_launcher_cmd(["ls"]), execute=execute, capture=True)
    if execute:
        print(ls_result.stdout, end="")
        if ls_result.returncode != 0:
            raise WorkflowError("runpod ls failed")
        if "No pods." in ls_result.stdout:
            return
        live_pod_ids = _listed_pod_ids(ls_result.stdout)
        allowed_pod_ids = allowed_pod_ids or set()
        if live_pod_ids and live_pod_ids.issubset(allowed_pod_ids):
            return
        if not args.allow_existing_pods:
            raise WorkflowError(f"{context}: stale RunPod pods exist; terminate or pass --allow-existing-pods deliberately")


def _launcher_train_args(
    *,
    model: ModelSpec,
    phase: PhaseSpec,
    job_id: str,
    manifest_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        "train",
        "--ffn-type",
        model.ffn_type,
        "--depth",
        str(model.depth),
        "--model-tag",
        model.tag,
        "--repo-ref",
        args.repo_ref,
        "--rational-kat-cu-ref",
        args.rational_kat_cu_ref,
        "--job-id",
        job_id,
        "--manifest-dir",
        str(manifest_dir),
    ]
    if model.ffn_type == "grkan" and model.grkan_groups is not None:
        cmd += ["--grkan-groups", str(model.grkan_groups)]
    if args.secure:
        cmd.append("--secure")
    if args.volume_id:
        cmd += ["--volume-id", args.volume_id]
    if phase.kind == "smoke":
        cmd.append("--smoke")
    else:
        cmd += ["--gate-approved", "--num-iterations", str(phase.num_iterations)]
        if phase.save_every:
            cmd += ["--save-every", str(phase.save_every)]
    if phase.max_runtime_minutes:
        cmd += ["--max-runtime-minutes", str(phase.max_runtime_minutes)]
    return cmd


def _mapped_artifact_path(artifact_dir: Path, expected: str) -> Path:
    if expected.startswith("base_checkpoints/"):
        return artifact_dir / expected.removeprefix("base_checkpoints/")
    return artifact_dir / expected


def _json_has_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_json_has_nonfinite(v) for v in value.values())
    if isinstance(value, list):
        return any(_json_has_nonfinite(v) for v in value)
    return False


def _verify_artifacts(phase_run: PhaseRun, model: ModelSpec, phase: PhaseSpec) -> None:
    artifact_dir = Path(phase_run.artifact_dir)
    launcher_manifest_path = Path(phase_run.manifest_path)
    if not launcher_manifest_path.exists():
        raise WorkflowError(f"missing launcher manifest: {launcher_manifest_path}")
    launcher_manifest = json.loads(launcher_manifest_path.read_text(encoding="utf-8"))
    if launcher_manifest.get("state") not in TERMINAL_OK_STATES:
        raise WorkflowError(f"launcher manifest is not complete: {launcher_manifest.get('state')}")

    for expected in launcher_manifest.get("expected_artifacts", []):
        path = _mapped_artifact_path(artifact_dir, expected)
        if "*" in str(path):
            if not list(path.parent.glob(path.name)):
                raise WorkflowError(f"missing expected artifact glob: {path}")
        elif not path.exists():
            raise WorkflowError(f"missing expected artifact: {path}")

    log_path = artifact_dir / f"{model.tag}_train.log"
    if not log_path.exists():
        raise WorkflowError(f"missing training log: {log_path}")
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    event = first_guard_event(lines)
    if event is not None:
        raise WorkflowError(f"guard event found in downloaded log: {event.reason}")
    if model.ffn_type == "grkan":
        log_text = "\n".join(lines)
        if "rational_kat_cu available" not in log_text:
            raise WorkflowError("GR-KAN log does not confirm rational_kat_cu availability")
        if "not installed" in log_text and "rational_kat_cu" in log_text:
            raise WorkflowError("GR-KAN log reports missing rational_kat_cu")

    final_step = phase.num_iterations
    if final_step is None:
        raise WorkflowError(f"phase {phase.name} has no final step")
    meta_path = artifact_dir / model.tag / f"meta_{final_step:06d}.json"
    model_path = artifact_dir / model.tag / f"model_{final_step:06d}.pt"
    optim_path = artifact_dir / model.tag / f"optim_{final_step:06d}_rank0.pt"
    for path in (meta_path, model_path, optim_path):
        if not path.exists():
            raise WorkflowError(f"missing resume-critical artifact: {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if _json_has_nonfinite(meta):
        raise WorkflowError(f"non-finite value found in metadata: {meta_path}")
    if meta.get("step") != final_step:
        raise WorkflowError(f"metadata step mismatch in {meta_path}: {meta.get('step')} != {final_step}")
    if not meta.get("dataloader_state_dict"):
        raise WorkflowError(f"metadata lacks dataloader resume state: {meta_path}")


def _completed_phase(manifest: WorkflowManifest, model_tag: str, phase_name: str) -> bool:
    for run in manifest.runs:
        if run["model_tag"] == model_tag and run["phase"] == phase_name:
            return run.get("state") == "verified"
    return False


def _find_phase_run(manifest: WorkflowManifest, model_tag: str, phase_name: str) -> PhaseRun | None:
    for run in manifest.runs:
        if run["model_tag"] == model_tag and run["phase"] == phase_name:
            return PhaseRun(**run)
    return None


def _record_run(manifest: WorkflowManifest, phase_run: PhaseRun) -> None:
    for idx, existing in enumerate(manifest.runs):
        if existing["job_id"] == phase_run.job_id:
            manifest.runs[idx] = asdict(phase_run)
            return
    manifest.runs.append(asdict(phase_run))


def _launcher_manifest_state(path: Path) -> str | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("state")


def _resume_allowed_pod_ids(manifest: WorkflowManifest) -> set[str]:
    pod_ids: set[str] = set()
    for run in manifest.runs:
        if run.get("state") == "verified":
            continue
        path = Path(run["manifest_path"])
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = payload.get("state")
        pod_id = payload.get("pod_id")
        if pod_id and state not in TERMINAL_OK_STATES and state not in TERMINAL_BAD_STATES:
            pod_ids.add(str(pod_id))
    return pod_ids


def _run_or_resume_phase(
    *,
    phase_run: PhaseRun,
    model: ModelSpec,
    phase: PhaseSpec,
    phase_manifest_dir: Path,
    args: argparse.Namespace,
    execute: bool,
) -> subprocess.CompletedProcess[str]:
    manifest_path = Path(phase_run.manifest_path)
    launcher_state = _launcher_manifest_state(manifest_path)
    if execute and launcher_state:
        if launcher_state in TERMINAL_OK_STATES:
            return subprocess.CompletedProcess(["verify-existing", str(manifest_path)], 0)
        if launcher_state in TERMINAL_BAD_STATES:
            raise WorkflowError(f"existing launcher manifest is terminal-bad: {launcher_state}")
        supervise_args = [
            "supervise",
            "--manifest",
            str(manifest_path),
            "--interval",
            "1",
            "--ssh-failure-limit",
            "6",
            "--terminate-on-done",
        ]
        return _run(_launcher_cmd(supervise_args), execute=True)

    train_args = _launcher_train_args(
        model=model,
        phase=phase,
        job_id=phase_run.job_id,
        manifest_dir=phase_manifest_dir,
        args=args,
    )
    return _run(_launcher_cmd(train_args), execute=execute)


def _run_workflow(args: argparse.Namespace, *, execute: bool, manifest_path: Path | None = None) -> Path:
    if manifest_path and manifest_path.exists():
        manifest = WorkflowManifest.load(manifest_path)
        workflow_id = manifest.workflow_id
        models = [ModelSpec(**spec) for spec in manifest.model_specs]
        phases = [PhaseSpec(**phase) for phase in manifest.phases]
    else:
        if not args.models:
            raise WorkflowError("--models is required for plan/run")
        models = _parse_models(args.models)
        phases = _phase_specs(args)
        workflow_id = args.workflow_id or _workflow_id(models, args.until)
        manifest_path = Path(args.workflow_dir) / f"{workflow_id}.json"
        manifest = WorkflowManifest(
            workflow_id=workflow_id,
            command=" ".join(sys.argv),
            repo_ref=args.repo_ref,
            rational_kat_cu_ref=args.rational_kat_cu_ref,
            volume_id=args.volume_id,
            secure=args.secure,
            model_specs=[asdict(m) for m in models],
            phases=[asdict(p) for p in phases],
            manifest_dir=str(args.manifest_dir),
            local_dest=str(args.local_dest),
        )
        manifest.save(manifest_path)

    manifest.event("preflight_start", execute=execute)
    manifest.save(manifest_path)
    allowed_pod_ids = _resume_allowed_pod_ids(manifest) if manifest_path and execute else None
    _preflight(args, phases, execute=execute, allowed_pod_ids=allowed_pod_ids)
    manifest.event("preflight_ok", execute=execute)
    manifest.state = "running" if execute else "planned"
    manifest.save(manifest_path)

    phase_manifest_dir = Path(args.manifest_dir)
    local_dest = Path(args.local_dest)
    for phase in phases:
        for model in models:
            if _completed_phase(manifest, model.tag, phase.name):
                print(f"SKIP: {model.tag} {phase.name} already verified")
                continue
            if execute and phase.requires_previous and not _completed_phase(manifest, model.tag, phase.requires_previous):
                raise WorkflowError(f"{model.tag} {phase.name} requires verified {phase.requires_previous}")

            existing_run = _find_phase_run(manifest, model.tag, phase.name)
            job_id = _phase_job_id(workflow_id, model, phase)
            phase_run = existing_run or PhaseRun(
                model_tag=model.tag,
                phase=phase.name,
                job_id=job_id,
                manifest_path=str(_phase_manifest_path(phase_manifest_dir, job_id)),
                artifact_dir=str(_artifact_dir(local_dest, job_id)),
            )
            phase_run.state = "running" if execute else "planned"
            phase_run.started_at = phase_run.started_at or (utc_now_iso() if execute else None)
            _record_run(manifest, phase_run)
            manifest.event("phase_start", model_tag=model.tag, phase=phase.name, job_id=job_id)
            manifest.save(manifest_path)

            result = _run_or_resume_phase(
                phase_run=phase_run,
                model=model,
                phase=phase,
                phase_manifest_dir=phase_manifest_dir,
                args=args,
                execute=execute,
            )
            phase_run.launcher_returncode = result.returncode
            if execute and result.returncode != 0:
                phase_run.state = "failed"
                phase_run.error = f"launcher exited {result.returncode}"
                _record_run(manifest, phase_run)
                manifest.state = "failed"
                manifest.event("phase_failed", model_tag=model.tag, phase=phase.name, error=phase_run.error)
                manifest.save(manifest_path)
                raise WorkflowError(phase_run.error)

            if execute:
                _verify_artifacts(phase_run, model, phase)
                _assert_no_pods(args, execute=True, context=f"post-{model.tag}-{phase.name}")
                phase_run.state = "verified"
                phase_run.completed_at = utc_now_iso()
            else:
                phase_run.state = "planned"
            _record_run(manifest, phase_run)
            manifest.event("phase_verified" if execute else "phase_planned", model_tag=model.tag, phase=phase.name)
            manifest.save(manifest_path)

    manifest.state = "complete" if execute else "planned"
    manifest.event("workflow_complete" if execute else "plan_complete")
    manifest.save(manifest_path)
    print(f"Workflow manifest: {manifest_path}")
    return manifest_path


def cmd_plan(args: argparse.Namespace) -> None:
    _run_workflow(args, execute=False)


def cmd_run(args: argparse.Namespace) -> None:
    _run_workflow(args, execute=True)


def cmd_resume(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise WorkflowError(f"manifest not found: {manifest_path}")
    stored = WorkflowManifest.load(manifest_path)
    args.models = ",".join(spec["tag"] for spec in stored.model_specs)
    args.repo_ref = stored.repo_ref
    args.rational_kat_cu_ref = stored.rational_kat_cu_ref
    args.volume_id = stored.volume_id
    args.secure = stored.secure
    args.until = stored.phases[-1]["name"]
    args.manifest_dir = stored.manifest_dir or args.manifest_dir
    args.local_dest = stored.local_dest or args.local_dest
    _run_workflow(args, execute=True, manifest_path=manifest_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe top-level RunPod workflow orchestrator")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models", default=None, help="Comma-separated model tags, e.g. d12-grkan-g4,d12-grkan-g16")
    common.add_argument("--repo-ref", default=None, help="Pinned nanokan commit; default is local HEAD")
    common.add_argument("--rational-kat-cu-ref", default="41a20b5", help="Pinned rational_kat_cu commit")
    common.add_argument("--volume-id", default=None, help="Network volume ID; required for pilot/full")
    common.add_argument("--secure", action="store_true", help="Use secure RunPod capacity")
    common.add_argument("--until", choices=["smoke", "pilot", "full"], default="pilot")
    common.add_argument("--pilot-steps", type=int, default=500)
    common.add_argument("--full-steps", type=int, default=2520)
    common.add_argument("--save-every", type=int, default=500)
    common.add_argument("--smoke-max-runtime-minutes", type=float, default=30)
    common.add_argument("--pilot-max-runtime-minutes", type=float, default=90)
    common.add_argument("--full-max-runtime-minutes", type=float, default=240)
    common.add_argument("--manifest-dir", default=str(DEFAULT_LOCAL_DEST / "runpod_manifests"))
    common.add_argument("--workflow-dir", default=str(DEFAULT_WORKFLOW_DIR))
    common.add_argument("--local-dest", default=str(DEFAULT_LOCAL_DEST))
    common.add_argument("--workflow-id", default=None)
    common.add_argument("--allow-existing-pods", action="store_true", help="Deliberately continue despite non-empty runpod ls")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", parents=[common], help="Validate and print the exact launcher sequence")
    sub.add_parser("run", parents=[common], help="Run the workflow with mandatory sequencing")
    resume = sub.add_parser("resume", parents=[common], help="Resume a workflow manifest")
    resume.add_argument("--manifest", required=True, help="Workflow manifest JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repo_ref is None:
        args.repo_ref = _repo_ref()
    try:
        if args.command == "plan":
            cmd_plan(args)
        elif args.command == "run":
            cmd_run(args)
        elif args.command == "resume":
            cmd_resume(args)
        else:  # pragma: no cover
            parser.error(f"unknown command {args.command}")
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
