"""
RunPod pipeline for nanochat GR-KAN PoC training.

Safe route: test → volume create → train baseline → train grkan → download → terminate

Subcommands
-----------
  test     [--gpu GPU] [--dry-run]
      Spin up a pod, clone the repo, install deps, run the GPU smoke test, auto-stop.
      Do this before every real training run (~5 min, ~$0.03).

  train    --ffn-type {mlp,grkan} [--depth N] [--gpu GPU] [--volume-id ID] [--dry-run]
      Launch a training pod. Pod stops itself when done (disk preserved).
      Use --volume-id to persist checkpoints across community-cloud interruptions.

  download <pod_id> [--dest DIR]
      rsync checkpoints from a stopped pod to local disk, then offer to terminate.
      Requires your SSH public key to be registered in RunPod settings.

  eval     [--model-tags TAGS] [--gpu GPU] [--dry-run]
      Launch an eval pod, rsync local checkpoints up, run CORE+BPB on each
      model tag, download result CSVs, terminate pod.
      TAGS is a comma-separated list (default: d12-grkan,d12-mlp).

  watch    <pod_id> [--dest DIR] [--interval MINUTES]
      Poll the pod every N minutes; auto-download when training finishes.

  ls
      List running/stopped pods with cost per hour.

  stop  <pod_id>
      Stop a pod (preserves disk) without terminating.

  terminate  <pod_id>
      Permanently terminate a pod and delete its disk.

  volume  create [--size GB]  |  ls  |  delete <volume_id>
      Manage persistent NVMe network volumes.
      Cost: ~$0.07/GB/month.  10 GB ≈ $0.70/month.
      Volumes survive pod terminations and community interruptions.
      Store tokenizer + dataset + checkpoints so the second run skips re-setup.

  gpus
      List available GPU types and community prices.

Prerequisites
-------------
    export RUNPOD_API_KEY=<your-key>       # RunPod console → Settings → API Keys
    export GITHUB_TOKEN=<your-pat>         # needed for train/test (repo clone)
    export WANDB_API_KEY=<your-key>        # optional but strongly recommended

IMPORTANT — Fork your nanochat before running
----------------------------------------------
Your GR-KAN changes are only on your local machine. RunPod needs to clone a
remote repo. Before using this script:

    1. Fork nanochat on GitHub (or push to any remote you own)
    2. Push your local changes:
           git remote set-url origin https://github.com/YOUR_USERNAME/nanochat
           git push -u origin main
    3. Update REPO_URL at the top of this file.

If you have not done this yet, `test` and `train` will clone karpathy/nanochat
(without your GR-KAN changes) and grkan training will fail.

CUDA version note
-----------------
nanochat requires torch 2.9.1 built for CUDA 12.8 (cu128).
This script uses `nvidia/cuda:12.8.1-cudnn9-devel-ubuntu22.04` as the base image.
The GPU driver on the RunPod host must be >= 570.xx (supports CUDA 12.8).
All community RTX 4090 and A6000 hosts with drivers from 2025+ satisfy this.

The -devel image is required (not -runtime) because rational_kat_cu compiles
a CUDA extension and needs nvcc.

Examples
--------
    # 1. Verify GPU + nanochat + rational_kat_cu (do this first, every time)
    python scripts/runpod_launch.py test

    # 2. Create a persistent volume once (reuse across both PoC runs)
    python scripts/runpod_launch.py volume create --size 10

    # 3. Train baseline (tokenizer + data setup happens automatically on first run)
    python scripts/runpod_launch.py train --ffn-type mlp --volume-id <id>

    # 4. Train GR-KAN (reuses tokenizer + data from volume, skips re-setup)
    python scripts/runpod_launch.py train --ffn-type grkan --volume-id <id>

    # 5. Monitor
    python scripts/runpod_launch.py ls

    # 6. Auto-download when training finishes (run in a separate terminal)
    python scripts/runpod_launch.py watch <pod_id>

    # 7. Or download manually once pod shows Exited
    python scripts/runpod_launch.py download <pod_id>
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import os
import subprocess
import sys
from pathlib import Path

try:
    import runpod
except ImportError:
    runpod = None


try:
    from scripts.runpod_supervisor import (
        GuardEvent,
        RunManifest,
        expected_training_artifacts,
        first_guard_event,
        is_pinned_git_ref,
        make_job_id,
        parse_log_line,
        require_pinned_git_ref,
        sha256_text,
        validate_train_request,
    )
except ImportError:  # when executed as python scripts/runpod_launch.py
    from runpod_supervisor import (  # type: ignore
        GuardEvent,
        RunManifest,
        expected_training_artifacts,
        first_guard_event,
        is_pinned_git_ref,
        make_job_id,
        parse_log_line,
        require_pinned_git_ref,
        sha256_text,
        validate_train_request,
    )

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_URL = "https://github.com/ACS-USP/nanokan"
BRANCH = "master"

# CUDA 12.8 devel image: required for nvcc to compile rational_kat_cu.
# Do NOT use -runtime — it lacks nvcc and rational_kat_cu will silently fall
# back to the pure-PyTorch Horner loop, which is ~123× slower on backward.
DOCKER_IMAGE = "nvidia/cuda:12.9.2-cudnn-devel-ubuntu22.04"

# H100 only for corrected GR-KAN scale runs. Non-H100 options are preserved below
# as commented fallback documentation, but are intentionally not active.
GPU_PREFERENCE = [
    "NVIDIA H100 80GB HBM3",          # H100 SXM — stock=High, US-NE-1
    "NVIDIA H100 PCIe",
    "NVIDIA A100-SXM4-80GB",          # 80 GB — valid for d12 per device_batch_size comment
    "NVIDIA A100 80GB PCIe",          # 80 GB — valid for d12 per device_batch_size comment
    # "NVIDIA GeForce RTX 4090",        # 24 GB ~$0.34/hr — prior best-value d12 option
    # "NVIDIA RTX PRO 4500 Blackwell",  # 32 GB ~$0.34/hr — EU-RO-1 available
    # "NVIDIA RTX A6000",               # 48 GB ~$0.33/hr
    # "NVIDIA RTX 5000 Ada Generation", # 32 GB ~$0.49/hr
    # "NVIDIA L40S",                    # 48 GB ~$0.79/hr
    # "NVIDIA L40",                     # 48 GB ~$0.69/hr
    # "NVIDIA A40",                     # 48 GB ~$0.49/hr
    # "NVIDIA RTX 6000 Ada Generation", # 48 GB ~$0.79/hr
    # "NVIDIA GeForce RTX 3090",        # 24 GB ~$0.24/hr
    # "NVIDIA GeForce RTX 3090 Ti",     # 24 GB ~$0.29/hr
    # "NVIDIA L4",                      # 24 GB ~$0.44/hr — EU-RO-1 available
]

VOLUME_MOUNT = "/runpod-volume"
# All nanochat artifacts (dataset shards, tokenizer, checkpoints) live here.
# Setting NANOCHAT_BASE_DIR to the volume means a second run reuses them.
NANOCHAT_CACHE_ON_VOLUME = f"{VOLUME_MOUNT}/nanochat"

# Number of ClimbMix shards to download for the PoC.
# d12 at --target-param-data-ratio=12 (~1.5B tokens) needs only ~8 shards,
# but 30 gives comfortable headroom for the training horizon and avoids
# cycling back through the same data too many times.
NUM_DATA_SHARDS = 30

# Save a checkpoint every N steps. Essential for community cloud (preemption risk).
# Use --resume-from-step <N> to restart from a saved checkpoint.
SAVE_EVERY = 500  # checkpoint every ~8 min on L40S; at most 8 min lost on preemption

# Sliding-window attention pattern.
# "L" (full context) for RTX 4090 / A6000 / L40S — PyTorch SDPA has no sliding
#    window support and will warn loudly if you use "SSSL" on these GPUs.
# "SSSL" for H100+ only (Flash Attention 3 supports sliding window natively).
WINDOW_PATTERN = "L"

# Default training depth for the PoC (matches the design doc).
DEFAULT_DEPTH = 12

LOCAL_DEST = "checkpoints/nanochat"
MANIFEST_DIR = "checkpoints/nanochat/runpod_manifests"
DEFAULT_RATIONAL_KAT_CU_REF = "41a20b5"  # fixed Safe Padé kernel commit recorded in wiki/log.md
DOWNLOAD_WINDOW_HOURS = 2
# Bound the time from SSH access until the training log first appears.  The in-pod
# watchdog only guards the *training* process, so a stall during pod setup (uv sync,
# rational_kat_cu build, tokenizer build) would otherwise hang the supervisor forever.
SETUP_DEADLINE_MINUTES = 25

# Community machines known to have persistent networking issues.
MACHINE_BLACKLIST: set[str] = {
    "3z47kcltj1d0",  # RTX 5000 Ada — cudaErrorDevicesUnavailable 2026-05-27
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_runpod():
    global runpod
    if runpod is not None:
        return runpod
    try:
        import runpod as runpod_module
    except ImportError:
        print("ERROR: runpod SDK not installed.  Run: uv add runpod")
        sys.exit(1)
    runpod = runpod_module
    return runpod

def get_api_key() -> str:
    rp = _require_runpod()
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        print("ERROR: RUNPOD_API_KEY not set.  Run: export RUNPOD_API_KEY=<your-key>")
        sys.exit(1)
    rp.api_key = key
    return key


def get_github_token() -> str:
    tok = os.environ.get("GITHUB_TOKEN", "")
    if not tok:
        print("ERROR: GITHUB_TOKEN not set.  Run: export GITHUB_TOKEN=<your-pat>")
        sys.exit(1)
    return tok


def _read_ssh_pubkey() -> str:
    for candidate in ["id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub"]:
        p = Path.home() / ".ssh" / candidate
        if p.exists():
            return p.read_text().strip()
    print("WARNING: No SSH public key found in ~/.ssh/. SSH access to pods will not work.")
    print("  Generate one with: ssh-keygen -t ed25519")
    return ""


def _try_launch_pod(
    name: str,
    gpu_type_id: str,
    startup: list[str],
    volume_id: str | None,
    disk_gb: int,
    env_extra: dict | None = None,
    secure: bool = False,
    image_name: str | None = None,
) -> dict | None:
    import time

    cmd = " && ".join(startup)
    # Only PYTHONUNBUFFERED here — no secrets in the env dict.
    # GITHUB_TOKEN is already embedded in the git clone URL in the startup script.
    # WANDB_API_KEY is base64-encoded into the startup script by _base_startup().
    # Passing secrets as env vars exposes them in plain text in the RunPod pod
    # details page; embedding them in docker_args keeps them out of that view.
    env: dict[str, str] = {"PYTHONUNBUFFERED": "1"}
    if env_extra:
        env.update(env_extra)

    kwargs = dict(
        name=name,
        image_name=image_name or DOCKER_IMAGE,
        gpu_type_id=gpu_type_id,
        cloud_type="SECURE" if secure else "COMMUNITY",
        container_disk_in_gb=disk_gb,
        start_ssh=True,
        support_public_ip=True,
        ports="22/tcp",
        env=env,
        docker_args=f"bash -c '{cmd}'",
    )
    if volume_id:
        kwargs["network_volume_id"] = volume_id
        kwargs["volume_mount_path"] = VOLUME_MOUNT

    pod = runpod.create_pod(**kwargs)
    pod_id = pod["id"]

    machine_id = ""
    for _ in range(8):
        time.sleep(5)
        info = runpod.get_pod(pod_id)
        machine_id = info.get("machineId") or ""
        if machine_id:
            break

    if machine_id in MACHINE_BLACKLIST:
        print(f"  WARNING: Landed on blacklisted machine {machine_id} — terminating …")
        runpod.terminate_pod(pod_id)
        time.sleep(10)
        raise RuntimeError(f"blacklisted:{machine_id}")

    return pod


def find_and_launch_pod(
    name: str,
    startup: list[str],
    preferred_gpu: str | None = None,
    volume_id: str | None = None,
    disk_gb: int = 60,
    env_extra: dict | None = None,
    secure: bool = False,
) -> tuple[str, dict]:
    # If a specific GPU is requested, only try that one — do not fall through to others.
    candidates = [preferred_gpu] if preferred_gpu else GPU_PREFERENCE
    print("Finding available GPU …")
    for gpu in candidates:
        try:
            pod = _try_launch_pod(name, gpu, startup, volume_id, disk_gb, env_extra, secure=secure)
            print(f"  Selected: {gpu}")
            return gpu, pod
        except RuntimeError as e:
            if str(e).startswith("blacklisted:"):
                print(f"  {gpu}: blacklisted machine — skipping")
            else:
                print(f"  {gpu}: ERROR — {e}")
        except Exception as e:
            msg = str(e)
            if "no longer any instances" in msg or "does not have the resources" in msg or "There are no longer" in msg:
                print(f"  {gpu}: unavailable")
            else:
                print(f"  {gpu}: ERROR — {e}")
    print("ERROR: No GPU from preference list is currently available.")
    sys.exit(1)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _local_git_ref(short: bool = True) -> str:
    root = _repo_root()
    args = ["git", "-C", str(root), "rev-parse"]
    if short:
        args.append("--short=12")
    args.append("HEAD")
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def _local_source_patch() -> bytes:
    root = _repo_root()
    paths = [
        "scripts/base_train.py",
        "scripts/runpod_gpu_test.py",
        "scripts/runpod_launch.py",
        "scripts/runpod_supervisor.py",
        "scripts/train_watchdog.py",
        "nanochat/gpt.py",
    ]
    return subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "--", *paths],
        check=True,
        capture_output=True,
    ).stdout


def _dirty_patch_sha256() -> str | None:
    root = _repo_root()
    chunks = [_local_source_patch()]
    for rel in ("scripts/runpod_supervisor.py", "scripts/train_watchdog.py"):
        path = root / rel
        if not path.exists():
            continue
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", rel],
            capture_output=True,
        ).returncode == 0
        if not tracked:
            chunks.append(f"\n--- untracked:{rel}\n".encode())
            chunks.append(path.read_bytes())
    payload = b"".join(chunks)
    return sha256_text(payload.decode("utf-8", errors="replace")) if payload else None


def _local_source_hotfix() -> list[str]:
    """Apply recorded local guardrail changes to the pod clone.

    This keeps short science-smoke iterations recoverable before the hardening
    patch is committed, while the manifest still records a patch hash.  New files
    are emitted explicitly because ``git diff`` omits untracked files.
    """
    root = _repo_root()
    cmds: list[str] = []
    diff = _local_source_patch()
    if diff:
        encoded = base64.b64encode(diff).decode()
        cmds += [
            f"echo {encoded} | base64 -d > /tmp/nanokan-local-hotfix.patch",
            "git apply /tmp/nanokan-local-hotfix.patch",
        ]
    for rel in ("scripts/runpod_supervisor.py", "scripts/train_watchdog.py"):
        path = root / rel
        if not path.exists():
            continue
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", rel],
            capture_output=True,
        ).returncode == 0
        if tracked:
            continue
        encoded = base64.b64encode(path.read_bytes()).decode()
        cmds.append(f"mkdir -p {Path(rel).parent}")
        cmds.append(f"echo {encoded} | base64 -d > {rel}")
    return cmds


def _base_startup(repo_ref: str | None = None, redact_secrets: bool = False) -> list[str]:
    """
    Common setup shared by test and train pods.

    Key differences from kanprey:
    - Base image has no Python/conda. We install python3 + python3-venv via apt.
    - Use uv (not pip directly) so the CUDA 12.8 torch index in pyproject.toml is honored.
    - `uv sync --extra gpu` installs torch 2.9.1 from download.pytorch.org/whl/cu128.
      Never use `pip install torch` here — it installs the CPU build from PyPI.
    - PATH must include uv's install location (~/.local/bin) before calling uv.
    """
    tok = "GITHUB_TOKEN_REDACTED" if redact_secrets else get_github_token()
    ssh_pubkey = "ssh-ed25519 REDACTED" if redact_secrets else _read_ssh_pubkey()
    wandb_key = "" if redact_secrets else os.environ.get("WANDB_API_KEY", "")

    steps = [
        "export DEBIAN_FRONTEND=noninteractive",
        "apt-get update -qq",
        # python3/python3-venv: needed by uv to create the .venv
        # python3-dev: Python.h headers required by Triton's CUDA backend (torch.compile)
        # git, rsync, openssh-server: repo clone + SSH access
        # curl: uv installer
        "apt-get install -y git rsync openssh-server python3 python3-venv python3-dev curl -qq",

        # SSH setup (docker_args overrides the container entrypoint, bypassing
        # RunPod's normal SSH injection; the CUDA base image has no sshd).
        "ssh-keygen -A",
        "mkdir -p /root/.ssh && chmod 700 /root/.ssh",
        # base64-encode the key to contain only [A-Za-z0-9+/=] — safe for
        # both GraphQL and single-quote bash context.
        f'echo {base64.b64encode(ssh_pubkey.encode()).decode()} | base64 -d > /root/.ssh/authorized_keys',
        "chmod 600 /root/.ssh/authorized_keys",
        # sshd_config drop-in: keywords must use key=value (no spaces in values)
        # to survive embedding in bash single-quotes.
        "mkdir -p /etc/ssh/sshd_config.d",
        "echo UsePAM=no > /etc/ssh/sshd_config.d/10-docker.conf",
        "echo PermitRootLogin=yes >> /etc/ssh/sshd_config.d/10-docker.conf",
        "echo StrictModes=no >> /etc/ssh/sshd_config.d/10-docker.conf",
        "echo PubkeyAuthentication=yes >> /etc/ssh/sshd_config.d/10-docker.conf",
        "mkdir -p /run/sshd",
        # Run sshd in foreground mode backgrounded by bash.
        # Plain daemon mode is unreliable inside Docker.
        "( /usr/sbin/sshd -D & ) && sleep 2",

        # Install uv (Python package manager).
        # uv installs to ~/.local/bin on Linux. Always export PATH before using it.
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH",

        # Clone nanokan (HCAI-USP/nanokan — the fork with GR-KAN changes).
        "rm -rf ~/nanokan",
        f"git clone --branch {BRANCH} https://{tok}@{REPO_URL.replace('https://', '')} ~/nanokan",
        "cd ~/nanokan",
        *( [f"git checkout {repo_ref}"] if repo_ref else [] ),
        *_local_source_hotfix(),

        # Install all dependencies using the project's lock + cu128 torch index.
        # `--extra gpu` selects the CUDA 12.8 build of torch from pyproject.toml.
        # Do NOT replace with `pip install torch` — that installs the CPU build.
        #
        # The venv is stored on the network volume so subsequent pod launches skip
        # the ~15-min download entirely: uv sync with an existing valid venv just
        # verifies package versions in ~5s and exits.  The local .venv is a symlink.
        "UV_PROJECT_ENVIRONMENT=/runpod-volume/nanochat/venv uv sync --extra gpu --quiet",
        "ln -sfn /runpod-volume/nanochat/venv .venv",
    ]

    # Embed WANDB_API_KEY via base64 so it stays out of the RunPod env vars table.
    # Same approach as the SSH pubkey: base64 contains only [A-Za-z0-9+/=], safe
    # inside bash single-quotes and the GraphQL API string.
    if wandb_key:
        encoded = base64.b64encode(wandb_key.encode()).decode()
        steps.append(f'export WANDB_API_KEY=$(echo {encoded} | base64 -d)')

    return steps


def _dataset_and_tokenizer_startup(nanochat_base: str) -> list[str]:
    """
    Download ClimbMix shards and train the tokenizer.
    Skipped if a sentinel file already exists on the volume (e.g. second run).

    IMPORTANT: the entire if/fi block is ONE list item so that when items are
    joined with ' && ' it doesn't produce `if ... then && body && fi` (invalid bash).
    Use semicolons to sequence commands inside the block instead of &&.
    """
    setup_block = (
        f"if [ ! -f {nanochat_base}/setup_done ]; then"
        # Download shards in parallel (4 workers). Shard 06542 is always the val shard.
        f" .venv/bin/python -m nanochat.dataset -n {NUM_DATA_SHARDS} -w 4"
        # Train the tokenizer on the first ~2B chars of data.
        # Tokenizer artifacts are saved inside NANOCHAT_BASE_DIR.
        " && .venv/bin/python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768"
        f" && touch {nanochat_base}/setup_done"
        "; fi"
    )
    return [
        f"export NANOCHAT_BASE_DIR={nanochat_base}",
        f"mkdir -p {nanochat_base}",
        # Symlink for easy rsync download later (regardless of where NANOCHAT_BASE_DIR is).
        # Re-run setup by deleting the sentinel: rm ~/nanochat_results/setup_done
        f"ln -sfn {nanochat_base} ~/nanochat_results",
        setup_block,
    ]


def _make_train_startup(
    depth: int,
    ffn_type: str,
    nanochat_base: str,
    run_name: str,
    model_tag_override: str | None = None,
    grkan_groups: int | None = None,
    smoke: bool = False,
    save_every_override: int | None = None,
    num_iterations: int | None = None,
    repo_ref: str | None = None,
    rational_kat_cu_ref: str = DEFAULT_RATIONAL_KAT_CU_REF,
    job_id: str | None = None,
    max_runtime_minutes: float | None = None,
    redact_secrets: bool = False,
) -> list[str]:
    rational_ref = require_pinned_git_ref(rational_kat_cu_ref, "rational_kat_cu_ref")
    cmds = _base_startup(repo_ref=repo_ref, redact_secrets=redact_secrets)

    # Install rational_kat_cu from a pinned commit. Unpinned HEAD installs are
    # forbidden for science runs because the fused Triton math is part of the result.
    # Cached: only reinstalls if the pinned ref has changed (sentinel on volume).
    # First install compiles Triton kernels (~5 min); subsequent launches are instant.
    #
    # NOTE: base64-encoded to avoid " and $(...) characters breaking the RunPod
    # GraphQL mutation string literal that embeds the docker_args startup command.
    rat_sentinel = "/runpod-volume/nanochat/rational_kat_cu_ref"
    # uv sync (earlier in startup) strips rational_kat_cu from the venv because
    # it is not in pyproject.toml. Add an importability check so we reinstall
    # whenever the package is missing, even when the version sentinel matches.
    rat_install_script = (
        f'if [ ! -f {rat_sentinel} ] || [ "$(cat {rat_sentinel})" != "{rational_ref}" ] \\\n'
        f'   || ! .venv/bin/python -c "import rational_kat_cu" 2>/dev/null; then\n'
        f'  uv pip install setuptools --quiet \\\n'
        f'  && uv pip install git+https://github.com/felippe-alves/rational_kat_cu.git@{rational_ref} --quiet \\\n'
        f'  && echo {rational_ref} > {rat_sentinel}\n'
        f'fi\n'
    )
    encoded_rat = base64.b64encode(rat_install_script.encode()).decode()
    cmds.append(f"echo {encoded_rat} | base64 -d > /tmp/install_rat.sh && bash /tmp/install_rat.sh")

    cmds += _dataset_and_tokenizer_startup(nanochat_base)

    model_tag = model_tag_override or f"d{depth}-{ffn_type}"
    log_file = f"{nanochat_base}/{model_tag}_train.log"
    runtime_manifest = {
        "job_id": job_id,
        "model_tag": model_tag,
        "repo_ref": repo_ref,
        "rational_kat_cu_ref": rational_ref,
        "nanochat_base": nanochat_base,
        "log_file": log_file,
    }
    encoded_manifest = base64.b64encode(json.dumps(runtime_manifest, indent=2).encode()).decode()
    cmds.append(f"echo {encoded_manifest} | base64 -d > {nanochat_base}/run_manifest.json")

    # Auto-resume: detect the last checkpoint inside this job directory and continue.
    # Job-specific volume paths prevent one experiment from resuming another run.
    ckpt_dir = f"{nanochat_base}/base_checkpoints/{model_tag}"
    resume_bash = (
        f"CKPT_DIR={ckpt_dir}\n"
        'LAST_STEP=$(\n'
        '  for model_path in "$CKPT_DIR"/model_*.pt; do\n'
        '    [ -e "$model_path" ] || continue\n'
        '    step=$(basename "$model_path" | sed "s/model_0*//" | sed "s/[.]pt//")\n'
        '    [ -f "$CKPT_DIR/meta_$(printf "%06d" "$step").json" ] || continue\n'
        '    [ -f "$CKPT_DIR/optim_$(printf "%06d" "$step")_rank0.pt" ] || continue\n'
        '    echo "$step"\n'
        '  done | sort -n | tail -1\n'
        ')\n'
        'if [ -n "$LAST_STEP" ] && [ "$LAST_STEP" -gt 0 ]; then\n'
        '  RESUME_FLAG="--resume-from-step $LAST_STEP"\n'
        '  echo "Resuming from step $LAST_STEP"\n'
        'else\n'
        '  RESUME_FLAG=""\n'
        '  echo "Starting from scratch"\n'
        'fi'
    )
    encoded_resume = base64.b64encode(resume_bash.encode()).decode()
    cmds.append(f"echo {encoded_resume} | base64 -d > /tmp/resume.sh")

    if save_every_override is not None:
        save_every = save_every_override
    elif smoke:
        save_every = 10
    else:
        save_every = SAVE_EVERY
    device_batch_size = 32  # needs >=50 GB VRAM; H100/A100 only for corrected d12 runs.
    debug_finite_steps = 100 if smoke else 20
    if smoke:
        extra_flags = " --num-iterations=20 --eval-every=10 --eval-tokens=524288 --sample-every=-1"
    elif num_iterations is not None:
        extra_flags = f" --num-iterations={num_iterations}"
    else:
        extra_flags = ""

    grkan_flags = ""
    if ffn_type == "grkan" and grkan_groups is not None:
        grkan_flags = f" --grkan-groups={grkan_groups}"

    runtime_flag = ""
    if max_runtime_minutes is not None and max_runtime_minutes > 0:
        runtime_flag = f" --max-runtime-minutes={max_runtime_minutes}"
    debug_artifact_dir = f"{nanochat_base}/failure/{model_tag}"
    train_cmd = (
        f".venv/bin/torchrun --standalone --nproc_per_node=1 -m scripts.base_train --"
        f" --depth={depth}"
        f" --ffn-type={ffn_type}"
        f" --model-tag={model_tag}"
        f"{grkan_flags}"
        f" --run={run_name}"
        f" --window-pattern={WINDOW_PATTERN}"
        f" --save-every={save_every}"
        f" --core-metric-every=-1"
        f" --device-batch-size={device_batch_size}"
        f" --fail-fast --debug-finite-steps={debug_finite_steps}"
        f" --debug-artifact-dir={debug_artifact_dir}"
        f"{runtime_flag}"
        f" $RESUME_FLAG"
        f"{extra_flags}"
    )
    cmds.append(f"echo Train command flags: --depth={depth} --ffn-type={ffn_type} --model-tag={model_tag}{grkan_flags} --fail-fast --debug-finite-steps={debug_finite_steps} --debug-artifact-dir={debug_artifact_dir}")
    train_script = f"set -euo pipefail\n. /tmp/resume.sh\n{train_cmd}\n"
    safe_model_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", model_tag)
    train_script_path = f"/tmp/train_{safe_model_tag}.sh"
    encoded_train = base64.b64encode(train_script.encode()).decode()
    cmds.append(f"echo {encoded_train} | base64 -d > {train_script_path} && chmod +x {train_script_path}")

    # If reusing a job dir (e.g. pilot → full run): archive the old log and remove
    # any stale DONE/FAILED sentinels.  The supervise reads from line 1, so an old
    # traceback at the end of the pilot log would trigger a false guard event; a
    # stale FAILED sentinel would make it stop the pod immediately.
    cmds.append(
        f"[ -f {log_file} ] && mv {log_file} {log_file}.$(date -u +%Y%m%dT%H%M%SZ).bak || true"
    )
    cmds.append(
        f"rm -f {nanochat_base}/DONE_* {nanochat_base}/FAILED_*"
    )

    watchdog_cmd = (
        f".venv/bin/python scripts/train_watchdog.py"
        f" --model-tag={model_tag}"
        f" --log {log_file}"
        f" --artifact-dir {nanochat_base}"
        f" --max-runtime-minutes={max_runtime_minutes if max_runtime_minutes else -1}"
        f" --heartbeat-timeout-minutes=10"
        f" --stop-pod-on-failure"
        f" -- bash {train_script_path}"
    )
    cmds.append(
        f"{watchdog_cmd}"
        f" || {{ touch {nanochat_base}/FAILED_{model_tag}; echo Training failed. Stopping pod to prevent further spend.; runpodctl stop pod $RUNPOD_POD_ID || true; sleep 600; }}"
    )

    cmds += [
        f"touch {nanochat_base}/DONE_{model_tag}",
        f"echo Training complete. Download immediately with: python scripts/runpod_launch.py download $RUNPOD_POD_ID",
        f"sleep {DOWNLOAD_WINDOW_HOURS * 3600}",
        "runpodctl terminate pod $RUNPOD_POD_ID",
    ]
    return cmds


# ── Subcommand: create-shell-template ────────────────────────────────────────

SHELL_IMAGE = "ubuntu:22.04"
SHELL_TEMPLATE_NAME = "nanochat-volume-shell"
# Stored alongside .env so the team can share it.
SHELL_TEMPLATE_ID_FILE = Path(__file__).resolve().parent.parent / ".runpod_shell_template_id"

def _shell_startup_cmds(ssh_pubkey: str) -> list[str]:
    """Startup commands for the shell pod.

    Avoids $VAR references (GraphQL reserved) and single-quotes (break bash -c '...').
    Uses base64-encoded key injection — same pattern as _base_startup().
    """
    b64key = base64.b64encode(ssh_pubkey.encode()).decode()
    return [
        "apt-get update -qq",
        "apt-get install -y openssh-server rsync curl -qq",
        "ssh-keygen -A",
        "mkdir -p /root/.ssh && chmod 700 /root/.ssh",
        f"echo {b64key} | base64 -d >> /root/.ssh/authorized_keys",
        "chmod 600 /root/.ssh/authorized_keys",
        "mkdir -p /etc/ssh/sshd_config.d",
        "echo PermitRootLogin=yes > /etc/ssh/sshd_config.d/10-docker.conf",
        "echo PubkeyAuthentication=yes >> /etc/ssh/sshd_config.d/10-docker.conf",
        "echo StrictModes=no >> /etc/ssh/sshd_config.d/10-docker.conf",
        "echo UsePAM=no >> /etc/ssh/sshd_config.d/10-docker.conf",
        "mkdir -p /run/sshd",
        "( /usr/sbin/sshd -D & ) && sleep 2",
        "touch /root/SHELL_READY",
        "sleep infinity",
    ]


def cmd_create_shell_template(_args):
    from runpod.api.graphql import run_graphql_query

    get_api_key()

    # Upsert: include the existing ID if we already created this template before.
    existing_id = SHELL_TEMPLATE_ID_FILE.read_text().strip() if SHELL_TEMPLATE_ID_FILE.exists() else ""
    # Template stores a generic placeholder key; real key is embedded at pod-launch time.
    # The template is used only to register the image+disk settings in the RunPod console.
    placeholder_b64 = base64.b64encode(b"PLACEHOLDER_KEY").decode()
    _template_cmds = [
        "apt-get update -qq",
        "apt-get install -y openssh-server rsync curl -qq",
        "ssh-keygen -A",
        "mkdir -p /root/.ssh && chmod 700 /root/.ssh",
        f"echo {placeholder_b64} | base64 -d >> /root/.ssh/authorized_keys",
        "chmod 600 /root/.ssh/authorized_keys",
        "mkdir -p /etc/ssh/sshd_config.d",
        "echo PermitRootLogin=yes > /etc/ssh/sshd_config.d/10-docker.conf",
        "echo PubkeyAuthentication=yes >> /etc/ssh/sshd_config.d/10-docker.conf",
        "echo StrictModes=no >> /etc/ssh/sshd_config.d/10-docker.conf",
        "echo UsePAM=no >> /etc/ssh/sshd_config.d/10-docker.conf",
        "mkdir -p /run/sshd",
        "( /usr/sbin/sshd -D & ) && sleep 2",
        "touch /root/SHELL_READY",
        "sleep infinity",
    ]
    docker_cmd_escaped = " && ".join(_template_cmds).replace('"', '\\"')
    id_field = f'id: "{existing_id}", ' if existing_id else ""
    mutation = f"""
    mutation {{
        saveTemplate(input: {{
            {id_field}
            name: "{SHELL_TEMPLATE_NAME}",
            imageName: "{SHELL_IMAGE}",
            dockerArgs: "{docker_cmd_escaped}",
            containerDiskInGb: 50,
            volumeInGb: 0,
            ports: "22/tcp",
            env: [],
            isServerless: false,
            startSsh: true,
            isPublic: false,
            readme: ""
        }}) {{ id name containerDiskInGb }}
    }}
    """
    result = run_graphql_query(mutation)
    tid = result["data"]["saveTemplate"]["id"]
    SHELL_TEMPLATE_ID_FILE.write_text(tid + "\n")
    action = "Updated" if existing_id else "Created"
    print(f"Template {action.lower()} : {SHELL_TEMPLATE_NAME}")
    print(f"Template ID      : {tid}")
    print(f"Saved to         : {SHELL_TEMPLATE_ID_FILE}")
    print(f"Console          : https://www.runpod.io/console/pods?templateId={tid}")
    print()
    print("Deploy via CLI:")
    print(f"  python scripts/runpod_launch.py shell --volume-id <VOLUME_ID>")


# ── Subcommand: shell ─────────────────────────────────────────────────────────

def cmd_shell(args):
    """Spin up a lightweight maintenance pod with a network volume mounted.

    Without --cmd: waits for SSH, prints connection string, and blocks until
    you press Ctrl-C (pod is stopped on exit).

    With --cmd: runs the command over SSH, prints output, then stops the pod.
    """
    import time

    get_api_key()

    # Resolve template ID.
    template_id = getattr(args, "template_id", None)
    if not template_id and SHELL_TEMPLATE_ID_FILE.exists():
        template_id = SHELL_TEMPLATE_ID_FILE.read_text().strip()
    if not template_id:
        print("No template ID found. Run first:")
        print("  python scripts/runpod_launch.py create-shell-template")
        sys.exit(1)

    ssh_pubkey = _read_ssh_pubkey()

    # H100 required — network volume xj62cpzdmv is only mountable on H100 datacenter nodes.
    shell_gpu_preference = [
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H100 PCIe",
    ]
    gpu_candidates = [args.gpu] if args.gpu else shell_gpu_preference

    # Startup: embed the real SSH key at launch time (avoids $VAR in GraphQL).
    startup_cmds = _shell_startup_cmds(ssh_pubkey)

    pod = None
    selected_gpu = None
    print("Launching shell pod …")
    for secure in (False, True):   # try community first, then secure cloud
        for gpu in gpu_candidates:
            try:
                pod = _try_launch_pod(
                    name="nanochat-shell",
                    gpu_type_id=gpu,
                    startup=startup_cmds,
                    volume_id=args.volume_id,
                    disk_gb=50,
                    secure=secure,
                    image_name=SHELL_IMAGE,
                )
                selected_gpu = f"{gpu} ({'secure' if secure else 'community'})"
                break
            except Exception as e:
                print(f"  {gpu} ({'secure' if secure else 'community'}): {e}")
                continue
        if pod is not None:
            break

    if pod is None:
        print("ERROR: could not launch shell pod on any GPU.")
        sys.exit(1)

    pod_id = pod["id"]
    print(f"  Pod      : {pod_id}  ({selected_gpu})")
    print(f"  Volume   : {args.volume_id} → {VOLUME_MOUNT}")
    print(f"  Console  : https://www.runpod.io/console/pods/{pod_id}")

    ssh_ip, ssh_port = _get_ssh_details(pod_id)
    _wait_for_ssh(ssh_ip, ssh_port)
    _wait_for_sentinel(ssh_ip, ssh_port, "/root/SHELL_READY", label="shell setup")

    ssh_e = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p {ssh_port}"

    if args.cmd:
        print(f"\nRunning: {args.cmd}\n{'─' * 60}")
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-p", str(ssh_port), f"root@{ssh_ip}", args.cmd],
            text=True,
        )
        print(f"{'─' * 60}\nExit code: {result.returncode}")
        print(f"\nStopping pod {pod_id} …")
        runpod.stop_pod(pod_id)
        print("Done.")
    else:
        print(f"\nShell ready. Connect with:")
        print(f"  ssh -o StrictHostKeyChecking=no -p {ssh_port} root@{ssh_ip}")
        print(f"\nVolume is at: {VOLUME_MOUNT}")
        print("Press Ctrl-C to stop the pod when finished.\n")
        try:
            while True:
                time.sleep(30)
        except KeyboardInterrupt:
            print(f"\nStopping pod {pod_id} …")
            runpod.stop_pod(pod_id)
            print("Done.")


# ── Subcommand: gpus ──────────────────────────────────────────────────────────

def cmd_gpus(_args):
    get_api_key()
    gpus = runpod.get_gpus()
    print(f"{'GPU':<44} {'VRAM':>6}  {'Community $/hr':>14}")
    print("-" * 72)
    for g in sorted(gpus, key=lambda x: x.get("communityPrice") or 999):
        mem = g.get("memoryInGb", "?")
        price = g.get("communityPrice", "?")
        price_str = f"${price}" if price != "?" else "n/a"
        print(f"{g['id']:<44} {str(mem)+'GB':>6}  {price_str:>14}")


# ── Subcommand: ls ────────────────────────────────────────────────────────────

def cmd_ls(_args):
    get_api_key()
    pods = runpod.get_pods()
    if not pods:
        print("No pods.")
        return
    print(f"{'Name':<35} {'ID':<20} {'GPU':<22} {'$/hr':>5}  {'Status'}")
    print("-" * 90)
    for p in pods:
        gpu = p.get("machine", {}).get("gpuDisplayName", "?")
        cost = p.get("costPerHr", "?")
        status = p.get("desiredStatus", "?")
        print(f"{p['name']:<35} {p['id']:<20} {gpu:<22} ${cost:>4}  {status}")


# ── Subcommand: stop ──────────────────────────────────────────────────────────

def cmd_stop(args):
    get_api_key()
    runpod.stop_pod(args.pod_id)
    print(f"Pod {args.pod_id} stopped (disk preserved).")


# ── Subcommand: terminate ─────────────────────────────────────────────────────

def cmd_terminate(args):
    get_api_key()
    runpod.terminate_pod(args.pod_id)
    print(f"Pod {args.pod_id} terminated.")


# ── Subcommand: volume ────────────────────────────────────────────────────────

def cmd_volume(args):
    get_api_key()

    from runpod.api.graphql import run_graphql_query

    if args.volume_action == "create":
        size = args.size
        DATACENTER_ID = getattr(args, "datacenter", None) or "EU-RO-1"
        mutation = f"""
        mutation {{
          createNetworkVolume(input: {{
            name: "nanochat-checkpoints",
            size: {size},
            dataCenterId: "{DATACENTER_ID}"
          }}) {{
            id name size dataCenterId
          }}
        }}
        """
        result = run_graphql_query(mutation)
        vol = result["data"]["createNetworkVolume"]
        vol_id = vol["id"]
        monthly_cost = size * 0.07
        print(f"Created volume: {vol_id}")
        print(f"  Size        : {size} GB")
        print(f"  Datacenter  : {DATACENTER_ID}")
        print(f"  Monthly cost: ~${monthly_cost:.2f}  ($0.07/GB/month)")
        print(f"\nFirst run:  python scripts/runpod_launch.py train --ffn-type mlp  --volume-id {vol_id}")
        print(f"Second run: python scripts/runpod_launch.py train --ffn-type grkan --volume-id {vol_id}")
        print()
        print(f"NOTE: Pod must be launched in the same datacenter as the volume ({DATACENTER_ID}).")
        print("      The second run reuses the tokenizer and dataset shards from the volume,")
        print("      skipping the ~5 min setup step automatically.")

    elif args.volume_action == "ls":
        query = "{ myself { networkVolumes { id name size dataCenterId } } }"
        result = run_graphql_query(query)
        vols = result.get("data", {}).get("myself", {}).get("networkVolumes", [])
        if not vols:
            print("No network volumes.")
            return
        print(f"{'ID':<25} {'Name':<28} {'Size':>6}  {'Datacenter'}")
        print("-" * 72)
        for v in vols:
            print(f"{v['id']:<25} {v.get('name','?'):<28} {str(v.get('size','?'))+'GB':>6}  {v.get('dataCenterId','?')}")

    elif args.volume_action == "delete":
        mutation = f"""
        mutation {{
          deleteNetworkVolume(input: {{ id: "{args.volume_id}" }})
        }}
        """
        run_graphql_query(mutation)
        print(f"Volume {args.volume_id} deleted.")


# ── Subcommand: test ──────────────────────────────────────────────────────────

def cmd_test(args):
    get_api_key()
    rational_ref = require_pinned_git_ref(args.rational_kat_cu_ref, "rational_kat_cu_ref")
    repo_ref = args.repo_ref or _local_git_ref()

    # Bring the pod to a ready SSH state, then run the gate over SSH so stdout is
    # captured by this local command. If this process dies, the pod stops itself
    # after the fallback sleep.
    test_block = (
        f"uv pip install git+https://github.com/felippe-alves/rational_kat_cu.git@{rational_ref} --quiet"
        " && touch /root/nanokan/READY_FOR_GPU_TEST"
        " && echo Environment ready for GPU gate"
        " && sleep 3600;"
        " runpodctl stop pod $RUNPOD_POD_ID || true"
    )
    startup = _base_startup(repo_ref=repo_ref, redact_secrets=args.dry_run) + [test_block]

    if args.dry_run:
        print("Startup script (dry run):")
        print("\n".join(f"  {s}" for s in startup))
        return

    gpu, pod = find_and_launch_pod(
        name="nanochat-gpu-test",
        startup=startup,
        preferred_gpu=args.gpu,
        disk_gb=20,
        secure=getattr(args, "secure", False),
    )
    pod_id = pod["id"]
    print(f"\nLaunching test pod on {gpu} …")
    print(f"  Pod ID  : {pod_id}")
    print(f"  Console : https://www.runpod.io/console/pods/{pod_id}")

    ssh_ip, ssh_port = _get_ssh_details(pod_id)
    _wait_for_ssh(ssh_ip, ssh_port)
    _wait_for_sentinel(ssh_ip, ssh_port, "/root/nanokan/READY_FOR_GPU_TEST", label="GPU gate setup")

    print("\nRunning remote GPU gate:")
    gate = subprocess.run(
        [
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
            "-p", str(ssh_port), f"root@{ssh_ip}",
            "cd /root/nanokan && .venv/bin/python scripts/runpod_gpu_test.py",
        ],
        text=True,
        capture_output=True,
    )
    print(gate.stdout, end="")
    if gate.stderr:
        print(gate.stderr, end="")
    runpod.stop_pod(pod_id)
    print(f"\nPod {pod_id} stopped.")
    if gate.returncode != 0:
        sys.exit(gate.returncode)


# ── Subcommand: train ─────────────────────────────────────────────────────────

def cmd_train(args):
    get_api_key()

    base_tag = f"d{args.depth}-{args.ffn_type}"
    if args.ffn_type == "grkan" and args.grkan_groups != 8:
        base_tag = f"{base_tag}-g{args.grkan_groups}"
    model_tag = args.model_tag or base_tag
    run_name = model_tag
    smoke = getattr(args, "smoke", False)
    try:
        rational_ref = require_pinned_git_ref(args.rational_kat_cu_ref, "rational_kat_cu_ref")
        validate_train_request(smoke=smoke, volume_id=args.volume_id, rational_kat_cu_ref=rational_ref)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)
    if not smoke and not args.gate_approved:
        print("ERROR: non-smoke training requires --gate-approved after local/static, H100 environment, and 20-step d12 smoke gates pass")
        sys.exit(2)

    repo_ref = args.repo_ref or _local_git_ref()
    job_id = args.job_id or make_job_id(model_tag)
    nanochat_base = f"{VOLUME_MOUNT}/nanochat/jobs/{job_id}" if args.volume_id else f"${{HOME}}/.cache/nanochat/jobs/{job_id}"
    final_step = 20 if smoke else args.num_iterations
    max_runtime_minutes = args.max_runtime_minutes
    if max_runtime_minutes is None:
        max_runtime_minutes = 30 if smoke else 240
    manifest = RunManifest(
        job_id=job_id,
        model_tag=model_tag,
        gpu=args.gpu,
        cloud_type="SECURE" if getattr(args, "secure", False) else "COMMUNITY",
        volume_id=args.volume_id,
        repo_url=REPO_URL,
        repo_ref=repo_ref,
        rational_kat_cu_ref=rational_ref,
        command=" ".join(sys.argv),
        max_runtime_minutes=max_runtime_minutes,
        max_cost_usd=args.max_cost_usd,
        expected_artifacts=expected_training_artifacts(model_tag, final_step=final_step),
        local_dest=LOCAL_DEST,
        remote_artifact_dir=nanochat_base,
        dirty_patch_sha256=_dirty_patch_sha256(),
    )

    startup = _make_train_startup(
        depth=args.depth,
        ffn_type=args.ffn_type,
        nanochat_base=nanochat_base,
        run_name=run_name,
        redact_secrets=args.dry_run,
        smoke=smoke,
        save_every_override=getattr(args, "save_every", None),
        num_iterations=getattr(args, "num_iterations", None),
        model_tag_override=model_tag,
        grkan_groups=getattr(args, "grkan_groups", None),
        repo_ref=repo_ref,
        rational_kat_cu_ref=rational_ref,
        job_id=job_id,
        max_runtime_minutes=max_runtime_minutes,
    )

    if args.dry_run:
        print("Manifest preview (dry run):")
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        print("\nStartup script (dry run):")
        print("\n".join(f"  {s}" for s in startup))
        if args.volume_id:
            print(f"\nVolume {args.volume_id} → {VOLUME_MOUNT}")
        return

    # Rough time estimate: d12 at ratio=12 = ~2800 steps × 2.5s/step ≈ 2h on 4090
    manifest_path = Path(args.manifest_dir) / f"{job_id}.json"
    manifest.save(manifest_path)
    print(f"  Manifest: {manifest_path}")

    est_hours = 2.5 if args.depth <= 12 else 5.0
    if args.ffn_type == "grkan":
        est_hours *= 1.15  # ~15% overhead for rational activations (with CUDA kernel)

    print(f"\nLaunching training pod:")
    print(f"  Depth   : {args.depth}  (n_embd = {args.depth * 64})")
    print(f"  FFN type: {args.ffn_type}")
    print(f"  Run name: {run_name}  (wandb + checkpoint tag)")
    if args.ffn_type == "grkan":
        print(f"  GR-KAN groups: {args.grkan_groups}")
    print(f"  Window  : {WINDOW_PATTERN}")
    if args.volume_id:
        print(f"  Volume  : {args.volume_id} → {VOLUME_MOUNT}")
        print(f"  Remote artifacts: {nanochat_base}")
    else:
        print("  Volume  : none (smoke-only job-local pod disk)")
    print(f"  Job ID  : {job_id}")
    print(f"  Repo ref: {repo_ref}")
    print(f"  rational_kat_cu: {rational_ref}")
    print(f"  Est. time: ~{est_hours:.1f}h")

    secure = getattr(args, "secure", False)
    gpu, pod = find_and_launch_pod(
        name=f"nanochat-{run_name}",
        startup=startup,
        preferred_gpu=args.gpu,
        volume_id=args.volume_id,
        disk_gb=60,
        secure=secure,
    )
    pod_id = pod["id"]
    cost = pod.get("costPerHr", "?")
    est_cost = est_hours * float(cost) if isinstance(cost, (int, float)) else "?"
    manifest.set_pod(pod_id, gpu=gpu)
    manifest.save(manifest_path)

    print(f"  GPU     : {gpu}")
    print(f"\n  Pod ID  : {pod_id}")
    print(f"  Console : https://www.runpod.io/console/pods/{pod_id}")
    print(f"  Rate    : ${cost}/hr")
    if est_cost != "?":
        print(f"  Est. cost: ~${est_cost:.2f}")
    print()
    if not os.environ.get("WANDB_API_KEY"):
        print("  NOTE: WANDB_API_KEY not set. Training will log to wandb project 'nanochat'")
        print("  with run='dummy' (disabled). Set WANDB_API_KEY before launching for real logging.")
        print()
    print("Training is fail-fast guarded by an in-pod watchdog.")
    if args.volume_id:
        print("Checkpoints are on a job-specific volume path; no sibling run shares this directory.")
    print(f"\nMonitor and retrieve artifacts:")
    print(f"  python scripts/runpod_launch.py supervise --manifest {manifest_path}")
    print(f"Manual download while pod is RUNNING:")
    print(f"  python scripts/runpod_launch.py download {pod_id}")
    if getattr(args, "detach", False):
        print("\nDetached mode requested; this command will not supervise the pod.")
        print("Run the supervise command above from a reliable terminal before launching anything else.")
        return

    print("\nAuto-supervising this manifest now. Use --detach only when another supervisor is already running.")

    class _Args:
        pass

    supervise_args = _Args()
    supervise_args.manifest = str(manifest_path)
    supervise_args.dest = None
    supervise_args.interval = 1
    supervise_args.ssh_failure_limit = 6
    supervise_args.terminate_on_done = True
    supervise_args.setup_deadline_minutes = getattr(args, "setup_deadline_minutes", None)
    cmd_supervise(supervise_args)


# ── Subcommand: download ──────────────────────────────────────────────────────

def _get_ssh_details(pod_id: str) -> tuple[str, int]:
    import time, json

    pod = runpod.get_pod(pod_id)
    if not pod:
        print(f"ERROR: Pod {pod_id} not found.")
        sys.exit(1)

    desired = pod.get("desiredStatus", "")
    if desired != "RUNNING":
        print(f"ERROR: Pod is '{desired}', not RUNNING.")
        print("Training pods stay alive for 2 hours after finishing — check the console.")
        sys.exit(1)

    print("Getting SSH connection details ", end="", flush=True)
    ssh_port, ssh_ip = None, None
    for _ in range(24):  # 24 × 5s = 2 min; CUDA devel image needs >60s to expose SSH
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") or []
        for p in ports:
            private_port = p.get("privatePort")
            port_type = str(p.get("type", "")).lower()
            if str(private_port) == "22" or (port_type == "tcp" and p.get("publicPort")):
                ssh_port = p.get("publicPort")
                ssh_ip = p.get("ip")
                break
        if ssh_port and ssh_ip:
            break
        time.sleep(5)
        pod = runpod.get_pod(pod_id)
        print(".", end="", flush=True)
    print()

    if not ssh_port or not ssh_ip:
        print(f"ERROR: Could not find SSH port for pod {pod_id}.")
        print("Pod runtime info:")
        print(json.dumps((pod.get("runtime") or {}), indent=2))
        sys.exit(1)

    return ssh_ip, ssh_port


def cmd_download(args):
    import time

    get_api_key()
    ssh_ip, ssh_port = _get_ssh_details(args.pod_id)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    ssh_opt = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p {ssh_port}"
    remote = f"root@{ssh_ip}"
    print(f"SSH: {remote} (port {ssh_port})")

    print("Waiting for SSH daemon ", end="", flush=True)
    for attempt in range(18):  # up to 3 min
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             "-p", str(ssh_port), f"root@{ssh_ip}", "echo ok"],
            capture_output=True,
        )
        if result.returncode == 0:
            print(" ready.")
            break
        print(".", end="", flush=True)
        time.sleep(10)
    else:
        print("\nERROR: SSH daemon did not become ready. Pod NOT terminated.")
        print(f"Try manually: ssh -p {ssh_port} root@{ssh_ip}")
        sys.exit(1)

    # Checkpoints are at ~/nanochat_results/ (symlink to NANOCHAT_BASE_DIR)
    results_cmd = [
        "rsync", "-av", "--partial", "--progress", "-e", ssh_opt,
        f"{remote}:~/nanochat_results/base_checkpoints/", str(dest) + "/",
    ]
    print(f"\nRunning: {' '.join(results_cmd)}")
    result = subprocess.run(results_cmd)
    if result.returncode != 0:
        print(f"\nERROR: rsync of base_checkpoints/ failed (exit {result.returncode}). Pod NOT terminated.")
        print(f"Try manually: ssh -p {ssh_port} root@{ssh_ip}")
        sys.exit(1)

    # Also grab the tokenizer — needed for local eval (base_eval.py --model-tag).
    tok_dest = Path(dest) / "tokenizer"
    tok_cmd = [
        "rsync", "-av", "--partial", "--progress", "-e", ssh_opt,
        f"{remote}:~/nanochat_results/tokenizer/", str(tok_dest) + "/",
    ]
    print(f"\nRunning: {' '.join(tok_cmd)}")
    result = subprocess.run(tok_cmd)
    if result.returncode not in (0, 23):
        print(f"WARNING: tokenizer rsync exited {result.returncode} (non-fatal; needed for local eval)")

    # Also grab training logs (exit 23 = no matching files — not an error)
    logs_cmd = [
        "rsync", "-av", "--progress", "-e", ssh_opt,
        f"{remote}:~/nanochat_results/*_train.log", str(dest) + "/",
    ]
    print(f"\nRunning: {' '.join(logs_cmd)}")
    result = subprocess.run(logs_cmd)
    if result.returncode not in (0, 23):
        print(f"WARNING: log rsync exited {result.returncode} (non-fatal)")

    # Grab manifest and fail-fast diagnostics first-class; these are tiny and
    # explain why a stopped run should not be trusted as science evidence.
    manifest_cmd = [
        "rsync", "-av", "--partial", "-e", ssh_opt,
        f"{remote}:~/nanochat_results/run_manifest.json", str(dest) + "/",
    ]
    print(f"\nRunning: {' '.join(manifest_cmd)}")
    result = subprocess.run(manifest_cmd)
    if result.returncode not in (0, 23):
        print(f"WARNING: manifest rsync exited {result.returncode} (non-fatal)")

    failure_dest = Path(dest) / "failure"
    failure_cmd = [
        "rsync", "-av", "--partial", "-e", ssh_opt,
        f"{remote}:~/nanochat_results/failure/", str(failure_dest) + "/",
    ]
    print(f"\nRunning: {' '.join(failure_cmd)}")
    result = subprocess.run(failure_cmd)
    if result.returncode not in (0, 23):
        print(f"WARNING: failure-bundle rsync exited {result.returncode} (non-fatal)")

    print(f"\nCheckpoints saved to: {dest}/")

    terminate_policy = getattr(args, "terminate", None)
    if terminate_policy is None:
        try:
            answer = input("Terminate pod (delete disk permanently)? [y/N] ").strip().lower()
        except EOFError:
            answer = "y"
            print("y  (auto-terminated — no stdin)")
        terminate_policy = answer == "y"

    if terminate_policy:
        runpod.terminate_pod(args.pod_id)
        print(f"Pod {args.pod_id} terminated.")
    else:
        print(f"Pod kept. Terminate later with:")
        print(f"  python scripts/runpod_launch.py terminate {args.pod_id}")


# ── Subcommand: eval ──────────────────────────────────────────────────────────
# H100 only for the corrected GR-KAN campaign; keep eval hardware matched.
# Non-H100 eval options are preserved as commented fallback documentation.
EVAL_GPU_PREFERENCE = [
    "NVIDIA H100 PCIe",
    "NVIDIA H100 80GB HBM3",
    # "NVIDIA GeForce RTX 4090",        # 24 GB ~$0.34/hr
    # "NVIDIA RTX PRO 4500 Blackwell",  # 32 GB ~$0.34/hr
    # "NVIDIA GeForce RTX 3090",        # 24 GB ~$0.24/hr
    # "NVIDIA GeForce RTX 3090 Ti",     # 24 GB ~$0.29/hr
    # "NVIDIA RTX A5000",               # 24 GB ~$0.30/hr
    # "NVIDIA RTX A4500",               # 20 GB ~$0.28/hr
    # "NVIDIA RTX A6000",               # 48 GB ~$0.33/hr
    # "NVIDIA A40",                     # 48 GB ~$0.49/hr
    # "NVIDIA L40",                     # 48 GB ~$0.69/hr
    # "NVIDIA L40S",                    # 48 GB ~$0.79/hr
    # "NVIDIA RTX 6000 Ada Generation", # 48 GB ~$0.79/hr
    # "NVIDIA L4",                      # 24 GB ~$0.44/hr
    # "NVIDIA RTX 5000 Ada Generation", # 32 GB ~$0.49/hr
    # "NVIDIA RTX 4000 Ada Generation", # 20 GB ~$0.35/hr
    # "NVIDIA RTX PRO 4000 Blackwell",  # 24 GB
    # "NVIDIA A100 80GB PCIe",          # 80 GB ~$1.89/hr
    # "NVIDIA A100-SXM4-80GB",          # 80 GB ~$2.29/hr
]


def _make_eval_startup(nanochat_base: str, upload_tokenizer: bool = False) -> list[str]:
    """
    Pod startup for eval: base env + rational_kat_cu.

    If upload_tokenizer=True (caller will rsync tokenizer from local), skips
    tokenizer training and writes READY_FOR_EVAL immediately after env setup.

    If upload_tokenizer=False (no local tokenizer), downloads a minimum set of
    ClimbMix shards and trains the tokenizer before writing the sentinel — this
    adds ~5–8 min but produces the correct vocabulary.
    """
    cmds = _base_startup()
    cmds.append("uv pip install setuptools --quiet")
    cmds.append("uv pip install git+https://github.com/felippe-alves/rational_kat_cu.git --quiet")
    cmds += [
        f"export NANOCHAT_BASE_DIR={nanochat_base}",
        f"mkdir -p {nanochat_base}/base_checkpoints",
        f"ln -sfn {nanochat_base} ~/nanochat_results",
    ]
    if not upload_tokenizer:
        # Must use the same 30 shards as training to reproduce the exact vocabulary.
        # Fewer shards produce a different token distribution → wrong embedding lookup
        # → near-random outputs (confirmed: 5-shard tokenizer gave BPB=2.28 vs 0.856).
        tok_block = (
            f"if [ ! -f {nanochat_base}/tokenizer/tokenizer.pkl ]; then"
            f" .venv/bin/python -m nanochat.dataset -n {NUM_DATA_SHARDS} -w 4"
            f" && .venv/bin/python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768"
            "; fi"
        )
        cmds.append(tok_block)
    cmds += [
        f"touch {nanochat_base}/READY_FOR_EVAL",
        "echo Environment ready. Waiting for checkpoint rsync and eval trigger.",
        "sleep infinity",
    ]
    return cmds


def _wait_for_ssh(ssh_ip: str, ssh_port: int, timeout_attempts: int = 36) -> None:
    """Poll SSH until it accepts connections (up to ~6 min)."""
    import time
    print("Waiting for sshd ", end="", flush=True)
    for _ in range(timeout_attempts):
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             "-o", "BatchMode=yes",
             "-p", str(ssh_port), f"root@{ssh_ip}", "echo ok"],
            capture_output=True,
        )
        if r.returncode == 0:
            print(" ready.")
            return
        print(".", end="", flush=True)
        time.sleep(10)
    print("\nERROR: sshd never became ready.")
    sys.exit(1)


def _wait_for_sentinel(ssh_ip: str, ssh_port: int, sentinel_path: str,
                       label: str = "setup", timeout_attempts: int = 60,
                       poll_interval: int = 10) -> None:
    """Poll via SSH until sentinel_path exists on the pod.

    Uses short reconnecting SSH calls so a dropped connection just retries
    rather than killing the wait. Default: 60 attempts × 10 s = 10 min.
    For long evals pass timeout_attempts=360, poll_interval=30 (3 h).
    """
    import time
    print(f"Waiting for {label} ", end="", flush=True)
    for _ in range(timeout_attempts):
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-p", str(ssh_port), f"root@{ssh_ip}",
             f"test -f {sentinel_path} && echo yes || echo no"],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and r.stdout.strip() == "yes":
            print(" done.")
            return
        print(".", end="", flush=True)
        time.sleep(poll_interval)
    print(f"\nERROR: {label} sentinel never appeared.")
    sys.exit(1)


def _wait_for_eval_done_or_failed(
    ssh_ip: str,
    ssh_port: int,
    nanochat_base: str,
    tag: str,
    timeout_attempts: int = 360,
    poll_interval: int = 30,
) -> bool:
    """Wait for one eval tag to finish, returning False when it failed remotely."""
    import time

    done_path = f"{nanochat_base}/DONE_{tag}"
    failed_path = f"{nanochat_base}/FAILED_{tag}"
    print(f"Waiting for {tag} eval (polls every {poll_interval} s, up to {timeout_attempts * poll_interval // 60} min) ", end="", flush=True)
    for _ in range(timeout_attempts):
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-p", str(ssh_port), f"root@{ssh_ip}",
             f"if [ -f {done_path} ]; then echo done; elif [ -f {failed_path} ]; then echo failed; else echo running; fi"],
            capture_output=True, text=True,
        )
        status = r.stdout.strip() if r.returncode == 0 else "running"
        if status == "done":
            print(" done.")
            return True
        if status == "failed":
            print(" failed.")
            return False
        print(".", end="", flush=True)
        time.sleep(poll_interval)
    print(f"\nERROR: {tag} eval did not finish before timeout.")
    return False


def _download_eval_results(ssh_ip: str, ssh_port: int, ssh_e: str, remote: str,
                           nanochat_base: str, model_tags: list, results_dir: "Path",
                           local_ckpt_base: "Path", have_tokenizer: bool) -> None:
    """rsync eval CSVs, logs, and (optionally) tokenizer from pod to local disk."""
    print("\nDownloading results …")
    subprocess.run([
        "rsync", "-av", "--progress", "-e", ssh_e,
        f"{remote}:{nanochat_base}/base_eval/", str(results_dir) + "/",
    ])
    for tag in model_tags:
        subprocess.run([
            "rsync", "-av", "-e", ssh_e,
            f"{remote}:{nanochat_base}/{tag}_eval.log", str(results_dir) + "/",
        ])
    print(f"\nResults saved to: {results_dir}/")
    for csv_file in sorted(results_dir.glob("*.csv")):
        print(f"\n--- {csv_file.name} ---")
        print(csv_file.read_text())

    if not have_tokenizer:
        print("\nDownloading tokenizer …")
        tok_dest = local_ckpt_base / "tokenizer"
        r = subprocess.run([
            "rsync", "-av", "--partial", "--progress", "-e", ssh_e,
            f"{remote}:{nanochat_base}/tokenizer/", str(tok_dest) + "/",
        ])
        if r.returncode in (0, 23):
            print(f"  Tokenizer saved to: {tok_dest}/")
        else:
            print(f"  WARNING: tokenizer download exited {r.returncode} (non-fatal)")


def cmd_eval(args):
    """
    Launch a cheap eval pod, rsync local checkpoints up, run CORE+BPB on each
    model tag, download result CSVs, terminate pod.
    """
    import time

    get_api_key()

    model_tags = [t.strip() for t in args.model_tags.split(",")]
    # LOCAL_DEST already points to the directory that contains model-tag subdirs.
    # (The download command rsyncs pod's base_checkpoints/ directly into LOCAL_DEST/.)
    local_ckpt_base = Path(LOCAL_DEST)
    nanochat_base = "/root/.cache/nanochat"

    # Verify all requested checkpoints exist locally before spending money.
    missing = []
    for tag in model_tags:
        ckpt_dir = local_ckpt_base / tag
        if not ckpt_dir.exists() or not list(ckpt_dir.glob("model_*.pt")):
            missing.append(str(ckpt_dir))
    if missing:
        print("ERROR: The following checkpoint directories are missing or empty:")
        for m in missing:
            print(f"  {m}")
        print("Run 'python scripts/runpod_launch.py download <pod_id>' first.")
        sys.exit(1)

    # Check if we have the tokenizer locally (downloaded by an updated cmd_download).
    local_tokenizer = local_ckpt_base / "tokenizer"
    have_tokenizer = (local_tokenizer / "tokenizer.pkl").exists()
    if have_tokenizer:
        print("Local tokenizer found — will upload to pod (saves ~5 min setup).")
    else:
        print("No local tokenizer — pod will build it from ClimbMix (~5–8 min extra).")

    startup = _make_eval_startup(nanochat_base, upload_tokenizer=have_tokenizer)

    if args.dry_run:
        print("=== Eval pod startup script (dry-run) ===")
        print(" && \\\n".join(startup))
        return

    secure = getattr(args, "secure", False)
    preferred_gpu = args.gpu
    candidates = [preferred_gpu] if preferred_gpu else EVAL_GPU_PREFERENCE
    gpu_type, pod = None, None
    # Disk budget: repo+venv ~4GB, ClimbMix 30 shards ~12GB, tokenizer ~50MB,
    # 2× final checkpoint (756MB each) + meta = ~2GB. 30GB with headroom.
    print("Finding available GPU …")
    for gpu in candidates:
        try:
            p = _try_launch_pod("nanochat-eval", gpu, startup, volume_id=None,
                                disk_gb=30, env_extra={}, secure=secure)
            gpu_type, pod = gpu, p
            print(f"  Selected: {gpu}")
            break
        except Exception as e:
            print(f"  {gpu}: {e}")
    if pod is None:
        print("ERROR: No eval GPU available.")
        sys.exit(1)

    pod_id = pod["id"]
    print(f"\n  Pod ID  : {pod_id}")
    print(f"  Console : https://www.runpod.io/console/pods/{pod_id}")

    # Wait for SSH endpoint to appear.
    print("\nWaiting for SSH endpoint ", end="", flush=True)
    ssh_ip, ssh_port = None, None
    for _ in range(60):
        try:
            pod = runpod.get_pod(pod_id)
        except Exception:
            time.sleep(10)
            continue
        ports = (pod.get("runtime") or {}).get("ports") or []
        p22 = next((p for p in ports if p.get("privatePort") == 22), None)
        if p22:
            ssh_ip, ssh_port = p22["ip"], p22["publicPort"]
            print(f" {ssh_ip}:{ssh_port}")
            break
        print(".", end="", flush=True)
        time.sleep(10)
    if not ssh_ip:
        print("\nERROR: SSH port never appeared.")
        runpod.terminate_pod(pod_id)
        sys.exit(1)

    _wait_for_ssh(ssh_ip, ssh_port)
    _wait_for_sentinel(ssh_ip, ssh_port,
                       f"{nanochat_base}/READY_FOR_EVAL", "env setup")

    ssh_e = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p {ssh_port}"
    remote = f"root@{ssh_ip}"

    # Upload tokenizer if available locally.
    if have_tokenizer:
        print("\nUploading tokenizer …")
        r = subprocess.run([
            "rsync", "-av", "--partial", "-e", ssh_e,
            str(local_tokenizer) + "/",
            f"{remote}:{nanochat_base}/tokenizer/",
        ])
        if r.returncode != 0:
            print("ERROR: tokenizer rsync failed.")
            runpod.terminate_pod(pod_id)
            sys.exit(1)
        print("  tokenizer: uploaded.")

    # Upload only the final checkpoint for each model tag (eval doesn't need intermediates).
    print("\nUploading checkpoints (final step only) …")
    for tag in model_tags:
        ckpt_dir = local_ckpt_base / tag
        # Find the highest-numbered model_*.pt
        pts = sorted(ckpt_dir.glob("model_*.pt"))
        metas = sorted(ckpt_dir.glob("meta_*.json"))
        if not pts:
            print(f"ERROR: no model_*.pt found in {ckpt_dir}")
            runpod.terminate_pod(pod_id)
            sys.exit(1)
        final_pt   = pts[-1]
        final_meta = metas[-1] if metas else None

        dst_dir = f"{remote}:{nanochat_base}/base_checkpoints/{tag}/"
        subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                        "-p", str(ssh_port), remote, f"mkdir -p {nanochat_base}/base_checkpoints/{tag}"],
                       check=True)
        for f in ([final_pt] + ([final_meta] if final_meta else [])):
            r = subprocess.run(["rsync", "-av", "--partial", "--progress",
                                "-e", ssh_e, str(f), dst_dir])
            if r.returncode != 0:
                print(f"ERROR: rsync of {f.name} failed.")
                runpod.terminate_pod(pod_id)
                sys.exit(1)
        print(f"  {tag}: {final_pt.name} uploaded.")

    # Smoke check: quick BPB on first model to verify checkpoint + tokenizer load correctly.
    smoke_tag = model_tags[0]
    print(f"\nSmoke check: loading {smoke_tag} …")
    smoke_cmd = (
        f"export NANOCHAT_BASE_DIR={nanochat_base} && "
        f"cd ~/nanokan && "
        f".venv/bin/python -m scripts.base_eval --model-tag {smoke_tag} "
        f"--eval bpb --device-batch-size=4 --split-tokens=524288 2>&1 | tail -6"
    )
    r = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=60",
         "-p", str(ssh_port), remote, smoke_cmd],
        timeout=300,
    )
    if r.returncode != 0:
        print(f"\nERROR: smoke check failed (exit {r.returncode}). Pod left running for debug.")
        print(f"SSH: ssh -p {ssh_port} root@{ssh_ip}")
        sys.exit(1)
    print("Smoke check passed.\n")

    results_dir = Path(LOCAL_DEST) / "base_eval"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Write a shell script to the pod and launch it as a nohup background job.
    # This decouples the eval from the local SSH session — if the laptop sleeps or
    # the network drops, the eval keeps running on the pod. We then poll for
    # DONE_{tag} sentinels with short reconnecting SSH calls.
    eval_lines = [
        "#!/bin/bash",
        f"export NANOCHAT_BASE_DIR={nanochat_base}",
        "cd ~/nanokan",
        f"mkdir -p {nanochat_base}/base_eval",
    ]
    for tag in model_tags:
        eval_lines += [
            f"if .venv/bin/python -m scripts.base_eval --model-tag {tag} "
            f"--eval core,bpb --device-batch-size=16 "
            f">{nanochat_base}/{tag}_eval.log 2>&1; then",
            f"  csv=$(ls -t {nanochat_base}/base_eval/base_model_*.csv 2>/dev/null | head -1)",
            f"  if [ -n \"$csv\" ]; then mv \"$csv\" \"{nanochat_base}/base_eval/{tag}_$(basename \"$csv\")\"; fi",
            f"  touch {nanochat_base}/DONE_{tag}",
            "else",
            f"  touch {nanochat_base}/FAILED_{tag}",
            "fi",
        ]
    eval_script = "\n".join(eval_lines) + "\n"

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tf:
        tf.write(eval_script)
        local_script = tf.name

    subprocess.run(
        ["scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no",
         local_script, f"{remote}:/root/run_eval.sh"],
        check=True,
    )
    # Save pod connection info BEFORE launching so eval-download can recover
    # even if the nohup launch SSH call times out.
    import json as _json
    pod_info = {
        "pod_id": pod_id, "ssh_ip": ssh_ip, "ssh_port": ssh_port,
        "nanochat_base": nanochat_base, "model_tags": model_tags,
        "have_tokenizer": have_tokenizer,
    }
    pod_info_path = results_dir / "running_pod.json"
    pod_info_path.write_text(_json.dumps(pod_info, indent=2))
    print(f"Pod info saved → {pod_info_path}")
    print(f"If this script is interrupted, recover with:")
    print(f"  python scripts/runpod_launch.py eval-download {pod_id}\n")

    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=60",
         "-p", str(ssh_port), remote,
         "chmod +x /root/run_eval.sh && nohup /root/run_eval.sh >/root/run_eval_nohup.out 2>&1 &"],
        timeout=120, check=True,
    )
    print("Eval jobs launched as nohup background process on pod.")
    print(f"If this script is interrupted, recover with:")
    print(f"  python scripts/runpod_launch.py eval-download {pod_id}\n")

    # Poll for DONE_{tag} sentinels — each call is a short SSH connection,
    # so brief network blips won't abort the wait.
    eval_ok = True
    for tag in model_tags:
        eval_ok = _wait_for_eval_done_or_failed(ssh_ip, ssh_port, nanochat_base, tag) and eval_ok

    _download_eval_results(ssh_ip, ssh_port, ssh_e, remote, nanochat_base,
                           model_tags, results_dir, local_ckpt_base, have_tokenizer)

    pod_info_path.unlink(missing_ok=True)
    print("\nTerminating pod …")
    runpod.terminate_pod(pod_id)
    print(f"Pod {pod_id} terminated.")
    if not eval_ok:
        print("ERROR: One or more eval jobs failed. Check downloaded *_eval.log files.")
        sys.exit(1)


# ── Subcommand: eval-download ─────────────────────────────────────────────────

def cmd_eval_download(args):
    """
    Resume/recover an eval that was launched by 'eval' but whose local script died
    (laptop slept, SSH dropped, etc.). The eval jobs keep running on the pod.

    Reads pod connection info from checkpoints/nanochat/base_eval/running_pod.json
    (written by 'eval') or accepts explicit --ssh HOST:PORT override.
    """
    import json as _json, time

    get_api_key()

    results_dir = Path(LOCAL_DEST) / "base_eval"
    local_ckpt_base = Path(LOCAL_DEST)
    pod_info_path = results_dir / "running_pod.json"

    if pod_info_path.exists():
        info = _json.loads(pod_info_path.read_text())
        pod_id       = args.pod_id or info["pod_id"]
        nanochat_base = info["nanochat_base"]
        model_tags    = info["model_tags"]
        have_tokenizer = info.get("have_tokenizer", False)
    else:
        if not args.pod_id:
            print("ERROR: No running_pod.json found and no pod_id given.")
            print(f"  Expected: {pod_info_path}")
            sys.exit(1)
        pod_id = args.pod_id
        nanochat_base = "/root/.cache/nanochat"
        model_tags = [t.strip() for t in (args.model_tags or "d12-grkan,d12-mlp").split(",")]
        have_tokenizer = False

    # Re-resolve SSH endpoint from RunPod API (IP/port can change after restart).
    print(f"Resolving SSH endpoint for pod {pod_id} …")
    ssh_ip, ssh_port = None, None
    if args.ssh:
        host, port = args.ssh.rsplit(":", 1)
        ssh_ip, ssh_port = host, int(port)
    else:
        for _ in range(24):
            try:
                pod = runpod.get_pod(pod_id)
            except Exception:
                time.sleep(10)
                continue
            ports = (pod.get("runtime") or {}).get("ports") or []
            p22 = next((p for p in ports if p.get("privatePort") == 22), None)
            if p22:
                ssh_ip, ssh_port = p22["ip"], p22["publicPort"]
                break
            print(".", end="", flush=True)
            time.sleep(10)

    if not ssh_ip:
        print("\nERROR: Could not resolve SSH endpoint. Is the pod still running?")
        print(f"  Check: https://www.runpod.io/console/pods/{pod_id}")
        sys.exit(1)

    print(f"  SSH: {ssh_ip}:{ssh_port}")
    _wait_for_ssh(ssh_ip, ssh_port)

    ssh_e = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p {ssh_port}"
    remote = f"root@{ssh_ip}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Wait for any still-running models.
    eval_ok = True
    for tag in model_tags:
        done_path = f"{nanochat_base}/DONE_{tag}"
        failed_path = f"{nanochat_base}/FAILED_{tag}"
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-p", str(ssh_port), remote,
             f"if [ -f {done_path} ]; then echo done; elif [ -f {failed_path} ]; then echo failed; else echo running; fi"],
            capture_output=True, text=True,
        )
        status = r.stdout.strip() if r.returncode == 0 else "running"
        if status == "done":
            print(f"  {tag}: already done.")
        elif status == "failed":
            print(f"  {tag}: failed.")
            eval_ok = False
        else:
            print(f"  {tag}: still running — waiting (polls every 30 s, up to 3 h) …")
            eval_ok = _wait_for_eval_done_or_failed(ssh_ip, ssh_port, nanochat_base, tag) and eval_ok

    _download_eval_results(ssh_ip, ssh_port, ssh_e, remote, nanochat_base,
                           model_tags, results_dir, local_ckpt_base, have_tokenizer)

    pod_info_path.unlink(missing_ok=True)
    print("\nTerminating pod …")
    runpod.terminate_pod(pod_id)
    print(f"Pod {pod_id} terminated.")
    if not eval_ok:
        print("ERROR: One or more eval jobs failed. Check downloaded *_eval.log files.")
        sys.exit(1)


# ── Subcommand: watch ─────────────────────────────────────────────────────────

def cmd_watch(args):
    """Poll the pod every N minutes; auto-download when training finishes."""
    import time

    # Force line-buffered stdout so progress is visible even when output is
    # redirected to a file (e.g. run_in_background=True in the Claude harness).
    sys.stdout.reconfigure(line_buffering=True)

    get_api_key()
    pod_id = args.pod_id
    poll_minutes = args.interval
    dest = Path(args.dest)
    volume_backed = bool(getattr(args, "volume_backed", False))
    ssh_failure_limit = getattr(args, "ssh_failure_limit", 6)
    manifest_path = getattr(args, "manifest_path", None)
    manifest = RunManifest.load(manifest_path) if manifest_path else None

    def _save_manifest_state(state: str, note: str | None = None) -> None:
        if manifest is None or manifest_path is None:
            return
        manifest.transition(state, note=note)
        manifest.save(manifest_path)

    def _download_from_running(terminate: bool) -> None:
        class _Args:
            pass

        dl_args = _Args()
        dl_args.pod_id = pod_id
        dl_args.dest = str(dest)
        dl_args.terminate = terminate
        cmd_download(dl_args)

    def _stop_for_supervision_failure(reason: str) -> None:
        print(f"\n{reason}")
        _save_manifest_state("stopping", reason)
        try:
            runpod.stop_pod(pod_id)
        finally:
            _save_manifest_state("stopped", reason)

    print(f"Watching pod {pod_id}. Will auto-download when training finishes.")
    print(f"Polling every {poll_minutes} min. Keep this terminal open. Ctrl+C to cancel.")
    _save_manifest_state("ssh_wait", "watch started")

    print("\nWaiting for SSH ", end="", flush=True)
    ssh_ip, ssh_port = None, None
    for _ in range(60):
        try:
            pod = runpod.get_pod(pod_id)
        except Exception:
            time.sleep(10)
            continue
        if not pod or pod.get("desiredStatus") != "RUNNING":
            status = pod.get("desiredStatus") if pod else "gone"
            print(f"\nPod is no longer RUNNING (status: {status}).")
            sys.exit(1)
        ports = (pod.get("runtime") or {}).get("ports") or []
        p22 = next((p for p in ports if p.get("privatePort") == 22), None)
        if p22:
            ssh_ip, ssh_port = p22["ip"], p22["publicPort"]
            break
        print(".", end="", flush=True)
        time.sleep(10)

    if not ssh_ip:
        _stop_for_supervision_failure("SSH port never appeared")
        sys.exit(1)

    print(f"\nSSH endpoint: {ssh_ip}:{ssh_port}")

    print("Waiting for sshd ", end="", flush=True)
    for _ in range(30):
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             "-o", "BatchMode=yes",
             "-p", str(ssh_port), f"root@{ssh_ip}", "echo ok"],
            capture_output=True,
        )
        if r.returncode == 0:
            print(" ready.\n")
            break
        print(".", end="", flush=True)
        time.sleep(10)
    else:
        _stop_for_supervision_failure("sshd never became ready")
        sys.exit(1)

    # Sentinel written by startup script after training finishes.
    # Model tag unknown here (could be d12-mlp or d12-grkan), so check either.
    check_cmd = "if ls ~/nanochat_results/FAILED_* >/dev/null 2>&1; then echo failed; elif ls ~/nanochat_results/DONE_* >/dev/null 2>&1; then echo done; else echo running; fi"
    ssh_opt = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p {ssh_port}"
    ssh_failures = 0
    log_line_offset = 1  # next unread line in the remote training log (1-indexed)
    training_started = False
    setup_deadline_minutes = getattr(args, "setup_deadline_minutes", None) or SETUP_DEADLINE_MINUTES
    setup_deadline = time.monotonic() + setup_deadline_minutes * 60.0

    while True:
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-p", str(ssh_port), f"root@{ssh_ip}", check_cmd],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            ssh_failures += 1
            if ssh_failures >= ssh_failure_limit:
                print(f"[{time.strftime('%H:%M')}] SSH failed {ssh_failures} times — stopping pod to cap spend.")
                _save_manifest_state("stopping", "ssh failure threshold exceeded")
                runpod.stop_pod(pod_id)
                _save_manifest_state("stopped", "stopped after ssh failure")
                sys.exit(1)
            print(f"[{time.strftime('%H:%M')}] SSH check failed ({ssh_failures}/{ssh_failure_limit}), retrying in 1 min …")
            time.sleep(60)
            continue

        ssh_failures = 0

        # Stream new log lines before acting on the status.
        log_r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-p", str(ssh_port), f"root@{ssh_ip}",
             f"awk 'NR>={log_line_offset}' ~/nanochat_results/*_train.log 2>/dev/null"],
            capture_output=True, text=True,
        )
        if log_r.returncode == 0 and log_r.stdout.strip():
            new_lines = log_r.stdout.splitlines()
            log_line_offset += len(new_lines)
            for line in new_lines:
                print(f"  {line}")
                event = parse_log_line(line)
                if event and event.should_stop:
                    print(f"[{time.strftime('%H:%M')}] Guard event detected: {event.reason}.")
                    if manifest is not None:
                        # event.fields for guard_fail events contains "reason" (parsed from the
                        # RUNPOD_GUARD_FAIL log line), so strip it before the ** expansion to
                        # avoid "got multiple values for keyword argument 'reason'".
                        extra = {k: v for k, v in event.fields.items() if k not in ("reason", "line")}
                        manifest.record_event(event.kind, reason=event.reason, line=event.line, **extra)
                        manifest.save(manifest_path)
                    if not volume_backed:
                        print("  Pod-local artifacts: downloading failure evidence before stop.")
                        _download_from_running(terminate=False)
                    print("  Stopping pod now to prevent further spend.")
                    _save_manifest_state("stopping", event.reason)
                    runpod.stop_pod(pod_id)
                    _save_manifest_state("stopped", event.reason)
                    sys.exit(1)

        # Setup-phase deadline: a non-empty training log means the train script is
        # running, i.e. clone/uv-sync/kernel-build/tokenizer all finished.  Until then
        # nothing guards against a setup stall, so bound the wait and stop compute.
        if not training_started and log_line_offset > 1:
            training_started = True
            print(f"[{time.strftime('%H:%M')}] Training started; setup-phase deadline cleared.")
        if not training_started and time.monotonic() > setup_deadline:
            reason = (
                f"training did not start within {setup_deadline_minutes:.0f} min of SSH access — "
                "pod setup stalled (uv sync / kernel build / tokenizer build)"
            )
            print(f"[{time.strftime('%H:%M')}] {reason} — stopping pod to cap spend.")
            _stop_for_supervision_failure(reason)
            sys.exit(1)

        status = r.stdout.strip().splitlines()[-1]
        if status == "failed":
            print(f"[{time.strftime('%H:%M')}] Training failed on the pod.")
            if volume_backed:
                print("Artifacts are on a network volume; stopping pod before any large download.")
                _save_manifest_state("stopping", "remote FAILED sentinel")
                runpod.stop_pod(pod_id)
                _save_manifest_state("stopped", "remote FAILED sentinel")
            else:
                print("Downloading failure evidence before stopping pod-local disk job …\n")
                _download_from_running(terminate=False)
                _save_manifest_state("stopping", "remote FAILED sentinel")
                runpod.stop_pod(pod_id)
                _save_manifest_state("stopped", "remote FAILED sentinel")
            sys.exit(1)
        if status == "done":
            print(f"[{time.strftime('%H:%M')}] Training finished! Starting download …\n")
            _save_manifest_state("downloading", "DONE sentinel")
            _download_from_running(terminate=bool(getattr(args, "terminate_on_done", False)))
            if bool(getattr(args, "terminate_on_done", False)):
                _save_manifest_state("terminated", "DONE artifacts downloaded and pod terminated")
            else:
                _save_manifest_state("downloaded", "DONE artifacts downloaded")
            return

        print(f"[{time.strftime('%H:%M')}] Still training. Polling for new checkpoints every 60 s …")
        if volume_backed:
            # Checkpoints are on the network volume; no need to rsync them incrementally.
            # A blocking rsync of large checkpoint files (model + optimizer > 2 GB) would
            # prevent the DONE sentinel from being re-checked for 10+ minutes.  Just sleep.
            time.sleep(poll_minutes * 60)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            # Check every 60 s so new checkpoints arrive locally within ~1 min of being written,
            # not after the full poll interval.  Status (done/failed) is rechecked after poll_minutes.
            deadline = time.monotonic() + poll_minutes * 60
            while time.monotonic() < deadline:
                sync = subprocess.run(
                    [
                        "rsync", "-a", "--ignore-existing", "--partial",
                        "--include=*/",
                        "--include=model_*.pt",
                        "--include=optim_*_rank0.pt",
                        "--include=meta_*.json",
                        "--exclude=*",
                        "-e", ssh_opt,
                        f"root@{ssh_ip}:~/nanochat_results/base_checkpoints/",
                        str(dest) + "/",
                    ],
                    capture_output=True, text=True,
                )
                new_ckpts = [l.strip() for l in sync.stdout.splitlines() if l.strip().endswith(".pt")]
                if new_ckpts:
                    print(f"[{time.strftime('%H:%M')}]   Downloaded: {', '.join(new_ckpts)}")
                elif sync.returncode not in (0, 23, 24):
                    print(f"[{time.strftime('%H:%M')}]   WARNING: rsync exited {sync.returncode}: {sync.stderr.strip()[:120]}")
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(60, remaining))



def cmd_supervise(args):
    """Supervise one manifest-backed training job."""
    manifest = RunManifest.load(args.manifest)
    if not manifest.pod_id:
        print(f"ERROR: manifest {args.manifest} has no pod_id yet.")
        sys.exit(1)

    class _Args:
        pass

    watch_args = _Args()
    watch_args.pod_id = manifest.pod_id
    watch_args.dest = args.dest or str(Path(manifest.local_dest) / manifest.job_id)
    watch_args.interval = args.interval
    watch_args.volume_backed = bool(manifest.volume_id)
    watch_args.ssh_failure_limit = args.ssh_failure_limit
    watch_args.terminate_on_done = args.terminate_on_done
    watch_args.manifest_path = args.manifest
    watch_args.setup_deadline_minutes = getattr(args, "setup_deadline_minutes", None)
    cmd_watch(watch_args)


def cmd_download_many(args):
    """Download artifacts from multiple RUNNING pods with bounded concurrency."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    get_api_key()
    root_dest = Path(args.dest)
    root_dest.mkdir(parents=True, exist_ok=True)

    def _one(pod_id: str) -> tuple[str, bool]:
        class _Args:
            pass

        dl_args = _Args()
        dl_args.pod_id = pod_id
        dl_args.dest = str(root_dest / pod_id)
        dl_args.terminate = args.terminate
        cmd_download(dl_args)
        return pod_id, True

    failures = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_one, pod_id): pod_id for pod_id in args.pod_ids}
        for future in as_completed(futures):
            pod_id = futures[future]
            try:
                future.result()
                print(f"{pod_id}: download complete")
            except Exception as exc:
                failures += 1
                print(f"{pod_id}: download failed: {exc}")
    if failures:
        sys.exit(1)

# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RunPod pipeline for nanochat GR-KAN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("gpus", help="List available GPU types and prices")
    sub.add_parser("ls", help="List running/stopped pods")

    sub.add_parser("create-shell-template",
                   help="Create (once) the reusable nanochat-volume-shell RunPod template")

    p_shell = sub.add_parser("shell",
                              help="Launch a lightweight maintenance pod with a volume mounted")
    p_shell.add_argument("--volume-id", required=True, help="Network volume ID to mount")
    p_shell.add_argument("--template-id", default=None,
                         help="RunPod template ID (default: read from .runpod_shell_template_id)")
    p_shell.add_argument("--gpu", default=None,
                         help="Preferred GPU type (default: cheapest available)")
    p_shell.add_argument("--cmd", default=None,
                         help="Run this command over SSH then stop the pod")

    p_stop = sub.add_parser("stop", help="Stop a pod (preserves disk)")
    p_stop.add_argument("pod_id")

    p_term = sub.add_parser("terminate", help="Terminate a pod (deletes disk)")
    p_term.add_argument("pod_id")

    p_vol = sub.add_parser("volume", help="Manage persistent network volumes")
    vol_sub = p_vol.add_subparsers(dest="volume_action", required=True)


    p_vc = vol_sub.add_parser("create", help="Create a new volume")
    p_vc.add_argument("--size", type=int, default=10,
                      help="Size in GB (default: 10; covers dataset + tokenizer + checkpoints)")
    p_vc.add_argument("--datacenter", default="EU-RO-1",
                      help="RunPod datacenter ID (default: EU-RO-1). Pod must be in same DC.")
    vol_sub.add_parser("ls", help="List volumes")
    p_vd = vol_sub.add_parser("delete", help="Delete a volume")
    p_vd.add_argument("volume_id")

    p_test = sub.add_parser("test", help="Run GPU smoke test pod (~5 min, ~$0.03)")
    p_test.add_argument("--gpu", default=None, help="Force a specific GPU type")
    p_test.add_argument("--dry-run", action="store_true",
                        help="Print startup script without launching")
    p_test.add_argument("--secure", action="store_true",
                        help="Use secure cloud H100 capacity for the smoke test")
    p_test.add_argument("--repo-ref", default=None,
                        help="nanokan commit/ref to checkout on the pod (default: local HEAD)")
    p_test.add_argument("--rational-kat-cu-ref", default=DEFAULT_RATIONAL_KAT_CU_REF,
                        help=f"pinned rational_kat_cu commit (default: {DEFAULT_RATIONAL_KAT_CU_REF})")

    p_train = sub.add_parser("train", help="Launch a training pod")
    p_train.add_argument("--ffn-type", dest="ffn_type", required=True,
                         choices=["mlp", "grkan"],
                         help="FFN type: mlp (baseline) or grkan (GR-KAN)")
    p_train.add_argument("--model-tag", dest="model_tag", default=None,
                         help="Explicit checkpoint/model tag. Defaults to d{depth}-{ffn_type} or d{depth}-grkan-g{groups}.")
    p_train.add_argument("--grkan-groups", dest="grkan_groups", type=int, default=8,
                         help="GR-KAN rational group count (default: 8). Only used with --ffn-type grkan.")
    p_train.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                         help=f"Model depth (default: {DEFAULT_DEPTH})")
    p_train.add_argument("--gpu", default=None, help="Force a specific GPU type")
    p_train.add_argument("--volume-id", default=None,
                         help="Persistent volume ID. Strongly recommended for community cloud.")
    p_train.add_argument("--smoke", action="store_true",
                         help="Smoke-test mode: 20 steps, eval/save every 10 steps, pod-local disk allowed.")
    p_train.add_argument("--num-iterations", dest="num_iterations", type=int, default=None,
                         help="Override total training steps (default: auto from param-data ratio).")
    p_train.add_argument("--save-every", dest="save_every", type=int, default=None,
                         help="Override save-every (checkpoints per N steps). Overrides smoke default.")
    p_train.add_argument("--gate-approved", action="store_true",
                         help="required for non-smoke training after staged smoke/pilot gates pass")
    p_train.add_argument("--secure", action="store_true",
                         help="Use secure cloud (no preemption). Costs more but guaranteed uptime.")
    p_train.add_argument("--dry-run", action="store_true",
                         help="Print startup script without launching")
    p_train.add_argument("--repo-ref", default=None,
                         help="nanokan commit/ref to checkout on the pod (default: local HEAD)")
    p_train.add_argument("--rational-kat-cu-ref", default=DEFAULT_RATIONAL_KAT_CU_REF,
                         help=f"pinned rational_kat_cu commit (default: {DEFAULT_RATIONAL_KAT_CU_REF})")
    p_train.add_argument("--job-id", default=None,
                         help="explicit manifest/job ID (default: timestamp-model_tag)")
    p_train.add_argument("--manifest-dir", default=MANIFEST_DIR,
                         help=f"local manifest directory (default: {MANIFEST_DIR})")
    p_train.add_argument("--max-runtime-minutes", type=float, default=None,
                         help="watchdog runtime cap (default: 30 smoke, 240 full)")
    p_train.add_argument("--max-cost-usd", type=float, default=None,
                         help="budget cap recorded in the manifest for supervisor policy")
    p_train.add_argument("--detach", action="store_true",
                         help="launch only and do not auto-supervise; use only when another supervisor is already running")
    p_train.add_argument("--setup-deadline-minutes", dest="setup_deadline_minutes", type=float, default=None,
                         help=f"stop the pod if training has not started within N min of SSH access (default: {SETUP_DEADLINE_MINUTES})")

    p_dl = sub.add_parser("download", help="rsync results from a running/stopped pod")
    p_dl.add_argument("pod_id")
    p_dl.add_argument("--dest", default=LOCAL_DEST,
                      help=f"Local destination directory (default: {LOCAL_DEST})")
    p_dl_term = p_dl.add_mutually_exclusive_group()
    p_dl_term.add_argument("--terminate", dest="terminate", action="store_true",
                           help="terminate after successful download")
    p_dl_term.add_argument("--no-terminate", dest="terminate", action="store_false",
                           help="keep pod after successful download")
    p_dl.set_defaults(terminate=None)

    p_watch = sub.add_parser("watch", help="Auto-download when training finishes")
    p_watch.add_argument("pod_id")
    p_watch.add_argument("--dest", default=LOCAL_DEST,
                         help=f"Local destination directory (default: {LOCAL_DEST})")
    p_watch.add_argument("--interval", type=int, default=5,
                         help="Polling interval in minutes (default: 5)")
    p_watch.add_argument("--volume-backed", action="store_true",
                         help="on failure, stop immediately because artifacts are on network volume")
    p_watch.add_argument("--ssh-failure-limit", type=int, default=6,
                         help="consecutive SSH failures before stopping pod (default: 6)")
    p_watch.add_argument("--terminate-on-done", action="store_true",
                         default=True,
                         help="terminate after successful auto-download (default)")
    p_watch.add_argument("--keep-pod-on-done", dest="terminate_on_done", action="store_false",
                         help="keep pod disk after successful auto-download")
    p_watch.add_argument("--setup-deadline-minutes", dest="setup_deadline_minutes", type=float, default=None,
                         help=f"stop the pod if training has not started within N min of SSH access (default: {SETUP_DEADLINE_MINUTES})")

    p_supervise = sub.add_parser("supervise", help="Supervise a manifest-backed training job")
    p_supervise.add_argument("--manifest", required=True, help="local run manifest JSON")
    p_supervise.add_argument("--dest", default=None, help="local artifact destination (default: manifest local_dest/job_id)")
    p_supervise.add_argument("--interval", type=int, default=1, help="polling interval in minutes (default: 1)")
    p_supervise.add_argument("--ssh-failure-limit", type=int, default=6)
    p_supervise.add_argument("--terminate-on-done", action="store_true",
                             default=True,
                             help="terminate pod after artifacts are confirmed local (default)")
    p_supervise.add_argument("--keep-pod-on-done", dest="terminate_on_done", action="store_false",
                             help="keep pod disk after artifacts are confirmed local")
    p_supervise.add_argument("--setup-deadline-minutes", dest="setup_deadline_minutes", type=float, default=None,
                             help=f"stop the pod if training has not started within N min of SSH access (default: {SETUP_DEADLINE_MINUTES})")

    p_dlm = sub.add_parser("download-many", help="Download artifacts from multiple RUNNING pods concurrently")
    p_dlm.add_argument("pod_ids", nargs="+")
    p_dlm.add_argument("--dest", default=LOCAL_DEST,
                       help=f"root destination; each pod writes under dest/<pod_id> (default: {LOCAL_DEST})")
    p_dlm.add_argument("--concurrency", type=int, default=3)
    p_dlm.add_argument("--terminate", action="store_true",
                       help="terminate pods after successful downloads")

    p_eval = sub.add_parser("eval", help="Run CORE+BPB eval on local checkpoints via a pod")
    p_eval.add_argument("--model-tags", dest="model_tags", default="d12-grkan,d12-mlp",
                        help="Comma-separated model tags to evaluate (default: d12-grkan,d12-mlp)")
    p_eval.add_argument("--gpu", default=None, help="Force a specific GPU type (default: cheapest available)")
    p_eval.add_argument("--secure", action="store_true",
                        help="Use secure cloud (guaranteed availability, costs more)")
    p_eval.add_argument("--dry-run", action="store_true",
                        help="Print startup script without launching")

    p_evdl = sub.add_parser(
        "eval-download",
        help="Recover a running eval: wait for completion, download results, terminate pod",
    )
    p_evdl.add_argument("pod_id", nargs="?", default=None,
                        help="Pod ID (optional if running_pod.json exists)")
    p_evdl.add_argument("--ssh", default=None, metavar="HOST:PORT",
                        help="Override SSH endpoint (e.g. 1.2.3.4:12345)")
    p_evdl.add_argument("--model-tags", dest="model_tags", default=None,
                        help="Comma-separated model tags (only needed without running_pod.json)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "gpus": cmd_gpus,
        "ls": cmd_ls,
        "stop": cmd_stop,
        "terminate": cmd_terminate,
        "volume": cmd_volume,
        "test": cmd_test,
        "train": cmd_train,
        "create-shell-template": cmd_create_shell_template,
        "shell": cmd_shell,
        "download": cmd_download,
        "watch": cmd_watch,
        "supervise": cmd_supervise,
        "download-many": cmd_download_many,
        "eval": cmd_eval,
        "eval-download": cmd_eval_download,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
