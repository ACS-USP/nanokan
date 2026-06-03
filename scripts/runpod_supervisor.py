"""RunPod supervision primitives for safe nanochat training.

This module is deliberately importable without the RunPod SDK.  Unit tests exercise
its parser, manifest, and policy code locally; ``runpod_launch.py`` wires the small
RunPod-specific API surface around it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PINNED_GIT_REF_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
JOB_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")

VALID_STATES = (
    "planned",
    "launched",
    "ssh_wait",
    "running",
    "recovering",
    "stopping",
    "stopped",
    "downloading",
    "downloaded",
    "terminating",
    "terminated",
    "failed",
)

_STATE_INDEX = {state: idx for idx, state in enumerate(VALID_STATES)}

_GUARD_FAIL_RE = re.compile(r"\bRUNPOD_GUARD_FAIL\b(?P<fields>.*)$")
_TRAIN_LOSS_RE = re.compile(r"\bloss:\s*(?P<loss>[-+0-9.eE]+|nan|inf|-inf)\b", re.IGNORECASE)
_VAL_BPB_RE = re.compile(r"\bValidation bpb:\s*(?P<val>[-+0-9.eE]+|nan|inf|-inf)\b", re.IGNORECASE)
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)|ChildFailedError|RuntimeError:", re.IGNORECASE)
_CUDA_ERROR_RE = re.compile(r"CUDA error|device-side assert|out of memory|CUBLAS_STATUS|NCCL", re.IGNORECASE)


@dataclass(frozen=True)
class GuardEvent:
    """A machine-actionable signal found in a training log."""

    kind: str
    reason: str
    line: str
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def should_stop(self) -> bool:
        return self.kind in {"guard_fail", "nonfinite_loss", "nonfinite_validation", "traceback", "cuda_error"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_job_component(value: str) -> str:
    cleaned = JOB_ID_SAFE_RE.sub("-", value.strip())
    return cleaned.strip("-._") or "job"


def make_job_id(model_tag: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{safe_job_component(model_tag)}"


def is_pinned_git_ref(ref: str | None) -> bool:
    return bool(ref and PINNED_GIT_REF_RE.fullmatch(ref.strip()))


def require_pinned_git_ref(ref: str | None, label: str) -> str:
    if not is_pinned_git_ref(ref):
        raise ValueError(f"{label} must be a pinned 7-40 hex commit, got {ref!r}")
    return ref.strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_guard_fields(field_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in field_text.strip().split():
        if "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        fields[key] = raw_value.strip().strip('"')
    return fields


def parse_log_line(line: str) -> GuardEvent | None:
    """Classify one log line.  Returns None for normal progress lines."""

    guard = _GUARD_FAIL_RE.search(line)
    if guard:
        fields = parse_guard_fields(guard.group("fields"))
        return GuardEvent("guard_fail", fields.get("reason", "guard_fail"), line.rstrip(), fields)

    loss = _TRAIN_LOSS_RE.search(line)
    if loss:
        value = loss.group("loss").lower()
        if value in {"nan", "inf", "+inf", "-inf"}:
            return GuardEvent("nonfinite_loss", f"loss_{value}", line.rstrip(), {"loss": value})

    val = _VAL_BPB_RE.search(line)
    if val:
        value = val.group("val").lower()
        if value in {"nan", "inf", "+inf", "-inf"}:
            return GuardEvent("nonfinite_validation", f"validation_{value}", line.rstrip(), {"val_bpb": value})

    if _CUDA_ERROR_RE.search(line):
        return GuardEvent("cuda_error", "cuda_error", line.rstrip(), {})

    if _TRACEBACK_RE.search(line):
        return GuardEvent("traceback", "traceback", line.rstrip(), {})

    return None


def first_guard_event(lines: Iterable[str]) -> GuardEvent | None:
    for line in lines:
        event = parse_log_line(line)
        if event and event.should_stop:
            return event
    return None


def validate_train_request(*, smoke: bool, volume_id: str | None, rational_kat_cu_ref: str | None) -> None:
    """Fail before pod creation when a request violates safety policy."""

    require_pinned_git_ref(rational_kat_cu_ref, "rational_kat_cu_ref")
    if not smoke and not volume_id:
        raise ValueError("non-smoke training requires --volume-id; pod-local disk is allowed only for smoke tests")


@dataclass
class RunManifest:
    job_id: str
    model_tag: str
    gpu: str | None
    cloud_type: str
    volume_id: str | None
    repo_url: str
    repo_ref: str
    rational_kat_cu_ref: str
    command: str
    max_runtime_minutes: float | None
    max_cost_usd: float | None
    expected_artifacts: list[str]
    local_dest: str
    state: str = "planned"
    pod_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    events: list[dict[str, Any]] = field(default_factory=list)
    remote_artifact_dir: str | None = None
    dirty_patch_sha256: str | None = None

    def __post_init__(self) -> None:
        require_pinned_git_ref(self.rational_kat_cu_ref, "rational_kat_cu_ref")
        if self.state not in VALID_STATES:
            raise ValueError(f"invalid manifest state {self.state!r}")

    def transition(self, new_state: str, *, note: str | None = None) -> None:
        if new_state not in VALID_STATES:
            raise ValueError(f"invalid manifest state {new_state!r}")
        old_index = _STATE_INDEX[self.state]
        new_index = _STATE_INDEX[new_state]
        if new_index < old_index and new_state not in {"failed", "stopping", "stopped"}:
            raise ValueError(f"invalid state regression {self.state!r} -> {new_state!r}")
        self.state = new_state
        self.updated_at = utc_now_iso()
        self.events.append({"time": self.updated_at, "state": new_state, "note": note})

    def set_pod(self, pod_id: str, gpu: str | None = None) -> None:
        self.pod_id = pod_id
        if gpu:
            self.gpu = gpu
        self.transition("launched", note=f"pod_id={pod_id}")

    def record_event(self, kind: str, *, reason: str | None = None, line: str | None = None, **fields: Any) -> None:
        event = {"time": utc_now_iso(), "kind": kind}
        if reason:
            event["reason"] = reason
        if line:
            event["line"] = line.rstrip()
        event.update(fields)
        self.events.append(event)
        self.updated_at = event["time"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "pod_id": self.pod_id,
            "model_tag": self.model_tag,
            "gpu": self.gpu,
            "cloud_type": self.cloud_type,
            "volume_id": self.volume_id,
            "repo_url": self.repo_url,
            "repo_ref": self.repo_ref,
            "rational_kat_cu_ref": self.rational_kat_cu_ref,
            "command": self.command,
            "max_runtime_minutes": self.max_runtime_minutes,
            "max_cost_usd": self.max_cost_usd,
            "expected_artifacts": self.expected_artifacts,
            "local_dest": self.local_dest,
            "remote_artifact_dir": self.remote_artifact_dir,
            "dirty_patch_sha256": self.dirty_patch_sha256,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunManifest":
        return cls(
            job_id=data["job_id"],
            model_tag=data["model_tag"],
            gpu=data.get("gpu"),
            cloud_type=data.get("cloud_type", "COMMUNITY"),
            volume_id=data.get("volume_id"),
            repo_url=data["repo_url"],
            repo_ref=data.get("repo_ref", ""),
            rational_kat_cu_ref=data["rational_kat_cu_ref"],
            command=data["command"],
            max_runtime_minutes=data.get("max_runtime_minutes"),
            max_cost_usd=data.get("max_cost_usd"),
            expected_artifacts=list(data.get("expected_artifacts", [])),
            local_dest=data.get("local_dest", "checkpoints/nanochat"),
            state=data.get("state", "planned"),
            pod_id=data.get("pod_id"),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
            events=list(data.get("events", [])),
            remote_artifact_dir=data.get("remote_artifact_dir"),
            dirty_patch_sha256=data.get("dirty_patch_sha256"),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def expected_training_artifacts(model_tag: str, final_step: int | None = None) -> list[str]:
    artifacts = [
        "run_manifest.json",
        f"{model_tag}_train.log",
        "tokenizer/tokenizer.pkl",
        "tokenizer/token_bytes.pt",
    ]
    if final_step is not None:
        artifacts.extend(
            [
                f"base_checkpoints/{model_tag}/model_{final_step:06d}.pt",
                f"base_checkpoints/{model_tag}/meta_{final_step:06d}.json",
            ]
        )
    else:
        artifacts.extend(
            [
                f"base_checkpoints/{model_tag}/model_*.pt",
                f"base_checkpoints/{model_tag}/meta_*.json",
            ]
        )
    return artifacts


def validate_artifact_paths(paths: Iterable[str]) -> None:
    seen: set[str] = set()
    for raw in paths:
        path = raw.strip()
        if not path or path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"unsafe artifact path {raw!r}")
        if path in seen:
            raise ValueError(f"duplicate artifact path {raw!r}")
        seen.add(path)


def log_tail(path: Path, max_bytes: int = 8192) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode("utf-8", errors="replace")


def write_failure_bundle(
    *,
    artifact_dir: str | Path,
    model_tag: str,
    event: GuardEvent,
    log_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist minimal diagnostics before a pod is stopped."""

    artifact_dir = Path(artifact_dir)
    failure_dir = artifact_dir / "failure" / model_tag
    failure_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "time": utc_now_iso(),
        "model_tag": model_tag,
        "kind": event.kind,
        "reason": event.reason,
        "line": event.line,
        "fields": event.fields,
    }
    if extra:
        payload.update(extra)
    if log_path:
        log = Path(log_path)
        (failure_dir / "last_log_tail.txt").write_text(log_tail(log), encoding="utf-8")
        payload["log_path"] = str(log)
    out = failure_dir / "failure.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (artifact_dir / f"FAILED_{model_tag}").write_text(payload["reason"] + "\n", encoding="utf-8")
    return out


def runtime_budget_exceeded(start_time: float, max_runtime_minutes: float | None) -> bool:
    return max_runtime_minutes is not None and max_runtime_minutes > 0 and (time.monotonic() - start_time) > max_runtime_minutes * 60
