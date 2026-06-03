"""In-pod watchdog for nanochat training.

The watchdog is intentionally independent of the local laptop/SSH session.  It
owns the child training process, parses its log for failure signals, kills the
process on first guardrail violation, writes a failure bundle, and optionally
asks RunPod to stop the pod so GPU billing cannot continue unnoticed.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

try:  # works both as ``python scripts/train_watchdog.py`` and module import
    from scripts.runpod_supervisor import (
        GuardEvent,
        parse_log_line,
        runtime_budget_exceeded,
        write_failure_bundle,
    )
except ImportError:  # pragma: no cover - script execution path on pod
    from runpod_supervisor import (  # type: ignore
        GuardEvent,
        parse_log_line,
        runtime_budget_exceeded,
        write_failure_bundle,
    )


def _terminate_process_tree(proc: subprocess.Popen[object], timeout_seconds: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout_seconds
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _stop_runpod_pod() -> None:
    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    if not pod_id:
        print("WATCHDOG: RUNPOD_POD_ID missing; cannot auto-stop pod", flush=True)
        return
    try:
        subprocess.run(["runpodctl", "stop", "pod", pod_id], timeout=30, check=False)
    except Exception as exc:  # pragma: no cover - depends on pod image/runpodctl
        print(f"WATCHDOG: runpodctl stop failed: {exc}", flush=True)


def _strip_separator(argv: list[str]) -> list[str]:
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def run_watchdog(
    *,
    train_cmd: list[str],
    log_path: Path,
    artifact_dir: Path,
    model_tag: str,
    max_runtime_minutes: float | None,
    heartbeat_timeout_minutes: float,
    poll_seconds: float,
    stop_pod_on_failure: bool,
) -> int:
    if not train_cmd:
        raise ValueError("train command is empty")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    last_progress = start_time

    print(f"WATCHDOG: starting {' '.join(train_cmd)}", flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        proc = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None
        failure_event: GuardEvent | None = None

        while True:
            line = proc.stdout.readline()
            if line:
                last_progress = time.monotonic()
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
                event = parse_log_line(line)
                if event and event.should_stop:
                    failure_event = event
                    break
            elif proc.poll() is not None:
                break
            else:
                now = time.monotonic()
                if runtime_budget_exceeded(start_time, max_runtime_minutes):
                    failure_event = GuardEvent(
                        kind="runtime_budget",
                        reason="max_runtime_minutes_exceeded",
                        line=f"runtime exceeded {max_runtime_minutes} minutes",
                        fields={"max_runtime_minutes": str(max_runtime_minutes)},
                    )
                    break
                if heartbeat_timeout_minutes > 0 and (now - last_progress) > heartbeat_timeout_minutes * 60:
                    failure_event = GuardEvent(
                        kind="missing_heartbeat",
                        reason="missing_progress_heartbeat",
                        line=f"no training log progress for {heartbeat_timeout_minutes} minutes",
                        fields={"heartbeat_timeout_minutes": str(heartbeat_timeout_minutes)},
                    )
                    break
                time.sleep(poll_seconds)

        if failure_event is not None:
            print(
                f"RUNPOD_GUARD_FAIL reason={failure_event.reason} model_tag={model_tag} source=watchdog",
                flush=True,
            )
            _terminate_process_tree(proc)
            write_failure_bundle(
                artifact_dir=artifact_dir,
                model_tag=model_tag,
                event=failure_event,
                log_path=log_path,
                extra={"source": "train_watchdog", "returncode": proc.poll()},
            )
            if stop_pod_on_failure:
                _stop_runpod_pod()
            return 42

        returncode = proc.wait()
        if returncode != 0:
            event = GuardEvent(
                kind="process_exit",
                reason=f"train_exit_{returncode}",
                line=f"training command exited {returncode}",
                fields={"returncode": str(returncode)},
            )
            write_failure_bundle(
                artifact_dir=artifact_dir,
                model_tag=model_tag,
                event=event,
                log_path=log_path,
                extra={"source": "train_watchdog"},
            )
            if stop_pod_on_failure:
                _stop_runpod_pod()
            return returncode

    print("WATCHDOG: training command completed successfully", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-fast watchdog for nanochat RunPod training")
    parser.add_argument("--log", required=True, type=Path, help="Path to append training stdout/stderr")
    parser.add_argument("--artifact-dir", required=True, type=Path, help="Job artifact directory")
    parser.add_argument("--model-tag", required=True, help="Model tag for sentinels/failure bundle")
    parser.add_argument("--max-runtime-minutes", type=float, default=-1.0, help="Kill training after this wall-clock budget")
    parser.add_argument("--heartbeat-timeout-minutes", type=float, default=10.0, help="Kill training if no log progress appears")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Polling interval while waiting for log progress")
    parser.add_argument("--stop-pod-on-failure", dest="stop_pod_on_failure", action="store_true", default=False)
    parser.add_argument("--no-stop-pod-on-failure", dest="stop_pod_on_failure", action="store_false")
    parser.add_argument("train_cmd", nargs=argparse.REMAINDER, help="Command to run after --")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_cmd = _strip_separator(args.train_cmd)
    max_runtime = args.max_runtime_minutes if args.max_runtime_minutes > 0 else None
    return run_watchdog(
        train_cmd=train_cmd,
        log_path=args.log,
        artifact_dir=args.artifact_dir,
        model_tag=args.model_tag,
        max_runtime_minutes=max_runtime,
        heartbeat_timeout_minutes=args.heartbeat_timeout_minutes,
        poll_seconds=args.poll_seconds,
        stop_pod_on_failure=args.stop_pod_on_failure,
    )


if __name__ == "__main__":
    raise SystemExit(main())
