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
import os
import subprocess
import sys
from pathlib import Path

try:
    import runpod
except ImportError:
    runpod = None


# ── Configuration ─────────────────────────────────────────────────────────────
REPO_URL = "https://github.com/ACS-USP/nanokan"
BRANCH = "master"

# CUDA 12.8 devel image: required for nvcc to compile rational_kat_cu.
# Do NOT use -runtime — it lacks nvcc and rational_kat_cu will silently fall
# back to the pure-PyTorch Horner loop, which is ~123× slower on backward.
DOCKER_IMAGE = "nvidia/cuda:12.9.2-cudnn-devel-ubuntu22.04"

# GPU preference: cheapest 24 GB+ first. All support CUDA 12.8 (driver >= 570.xx).
GPU_PREFERENCE = [
    "NVIDIA H100 PCIe",               # 80 GB ~$2.49/hr — target for grkan full run
    "NVIDIA H100 80GB HBM3",          # 80 GB ~$2.99/hr — H100 SXM fallback
    "NVIDIA GeForce RTX 4090",        # 24 GB ~$0.34/hr — best value for d12
    "NVIDIA RTX PRO 4500 Blackwell",  # 32 GB ~$0.34/hr — EU-RO-1 available
    "NVIDIA RTX A6000",               # 48 GB ~$0.33/hr
    "NVIDIA RTX 5000 Ada Generation", # 32 GB ~$0.49/hr
    "NVIDIA L40S",                    # 48 GB ~$0.79/hr
    "NVIDIA L40",                     # 48 GB ~$0.69/hr
    "NVIDIA A40",                     # 48 GB ~$0.49/hr
    "NVIDIA RTX 6000 Ada Generation", # 48 GB ~$0.79/hr
    "NVIDIA GeForce RTX 3090",        # 24 GB ~$0.24/hr
    "NVIDIA GeForce RTX 3090 Ti",     # 24 GB ~$0.29/hr
    "NVIDIA L4",                      # 24 GB ~$0.44/hr — EU-RO-1 available
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
DOWNLOAD_WINDOW_HOURS = 2

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
) -> dict | None:
    import time

    tok = get_github_token()

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
        image_name=DOCKER_IMAGE,
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


def _base_startup() -> list[str]:
    """
    Common setup shared by test and train pods.

    Key differences from kanprey:
    - Base image has no Python/conda. We install python3 + python3-venv via apt.
    - Use uv (not pip directly) so the CUDA 12.8 torch index in pyproject.toml is honored.
    - `uv sync --extra gpu` installs torch 2.9.1 from download.pytorch.org/whl/cu128.
      Never use `pip install torch` here — it installs the CPU build from PyPI.
    - PATH must include uv's install location (~/.local/bin) before calling uv.
    """
    tok = get_github_token()
    ssh_pubkey = _read_ssh_pubkey()
    wandb_key = os.environ.get("WANDB_API_KEY", "")

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

        # Install all dependencies using the project's lock + cu128 torch index.
        # `--extra gpu` selects the CUDA 12.8 build of torch from pyproject.toml.
        # Do NOT replace with `pip install torch` — that installs the CPU build.
        "uv sync --extra gpu --quiet",
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
    smoke: bool = False,
    save_every_override: int | None = None,
    num_iterations: int | None = None,
) -> list[str]:
    cmds = _base_startup()

    # Install rational_kat_cu from our fork. The fork adds a `rational_kat_cu` package
    # that wraps the existing Triton kernels with the interface nanochat/gpt.py expects.
    # No CUDA compilation needed — Triton ships with PyTorch.
    cmds.append("uv pip install setuptools --quiet")
    cmds.append("uv pip install git+https://github.com/felippe-alves/rational_kat_cu.git --quiet")

    cmds += _dataset_and_tokenizer_startup(nanochat_base)

    model_tag = f"d{depth}-{ffn_type}"
    log_file = f"{nanochat_base}/{model_tag}_train.log"

    # Auto-resume: detect the last checkpoint on the volume and continue from there
    # instead of restarting from zero after a preemption. No-op on a fresh run.
    # The logic uses double quotes and $ expansions which are illegal in GraphQL strings,
    # so encode it as base64 and decode+source it on the pod at runtime.
    ckpt_dir = f"{nanochat_base}/base_checkpoints/{run_name}"
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
    cmds.append(f"echo {encoded_resume} | base64 -d > /tmp/resume.sh && . /tmp/resume.sh")

    if save_every_override is not None:
        save_every = save_every_override
    elif smoke:
        save_every = 10
    else:
        save_every = SAVE_EVERY
    # device_batch_size=32 needs ≥50 GB VRAM (H100/A100); grad_accum halved to 8 → fewer
    # graph-break dispatches per step. Falls back to 16 on smaller GPUs if needed.
    device_batch_size = 32
    if smoke:
        extra_flags = " --num-iterations=10"
    elif num_iterations is not None:
        extra_flags = f" --num-iterations={num_iterations}"
    else:
        extra_flags = ""

    train_cmd = (
        f"( set -o pipefail;"
        f" .venv/bin/torchrun --standalone --nproc_per_node=1 -m scripts.base_train --"
        f" --depth={depth}"
        f" --ffn-type={ffn_type}"
        f" --model-tag={model_tag}"
        f" --run={run_name}"
        f" --window-pattern={WINDOW_PATTERN}"
        f" --save-every={save_every}"
        f" --core-metric-every=-1"
        f" --device-batch-size={device_batch_size}"
        f" $RESUME_FLAG"
        f"{extra_flags}"
        f" 2>&1 | tee {log_file})"
    )
    # On failure: sleep infinity so the pod stays alive for SSH debugging.
    # exit 1 (or pipefail) would cause RunPod to restart the container in a loop.
    # The log at {log_file} contains the full Python traceback before the torchrun
    # ChildFailedError — check it first when debugging.
    cmds.append(
        f"{train_cmd}"
        f" || {{ touch {nanochat_base}/FAILED_{model_tag}; echo TRAINING FAILED — pod sleeping for debug. Check log: {log_file}; sleep infinity; }}"
    )

    # Keep pod alive after training for SSH download.
    # Pod auto-terminates after DOWNLOAD_WINDOW_HOURS if not terminated sooner.
    cmds += [
        f"touch {nanochat_base}/DONE_{model_tag}",
        f"echo Training complete. Pod will auto-terminate in {DOWNLOAD_WINDOW_HOURS}h.",
        f"echo Download now with: python scripts/runpod_launch.py download $RUNPOD_POD_ID",
        f"sleep {DOWNLOAD_WINDOW_HOURS * 3600}",
        "runpodctl terminate pod $RUNPOD_POD_ID",
    ]
    return cmds


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
        # EU-RO-1 has reliable RTX 4090 community availability.
        # Pod and volume MUST be in the same datacenter — see DATACENTER_ID note.
        DATACENTER_ID = "EU-RO-1"
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
        print("NOTE: Pod must be launched in the same datacenter as the volume (EU-RO-1).")
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
          deleteNetworkVolume(input: {{ id: "{args.volume_id}" }}) {{
            id
          }}
        }}
        """
        run_graphql_query(mutation)
        print(f"Volume {args.volume_id} deleted.")


# ── Subcommand: test ──────────────────────────────────────────────────────────

def cmd_test(args):
    get_api_key()

    # The stop command is placed outside the && chain via a bash trap so the pod
    # always stops itself even if uv pip install or the test script fails.
    # Without this, a failing step leaves the pod RUNNING and billing indefinitely.
    test_block = (
        "{ uv pip install git+https://github.com/felippe-alves/rational_kat_cu.git --quiet"
        " && .venv/bin/python scripts/runpod_gpu_test.py; };"
        " runpodctl stop pod $RUNPOD_POD_ID || true"
    )
    startup = _base_startup() + [test_block]

    if args.dry_run:
        print("Startup script (dry run):")
        print("\n".join(f"  {s}" for s in startup))
        return

    gpu, pod = find_and_launch_pod(
        name="nanochat-gpu-test",
        startup=startup,
        preferred_gpu=args.gpu,
        disk_gb=20,
    )
    print(f"\nLaunching test pod on {gpu} …")
    pod_id = pod["id"]
    print(f"  Pod ID  : {pod_id}")
    print(f"  Console : https://www.runpod.io/console/pods/{pod_id}")
    print()
    print("The pod runs the smoke test and stops itself (~5 min, ~$0.03).")
    print("Watch the pod logs in the RunPod console for PASS/FAIL output.")
    print("Once stopped, terminate with:")
    print(f"  python scripts/runpod_launch.py terminate {pod_id}")


# ── Subcommand: train ─────────────────────────────────────────────────────────

def cmd_train(args):
    get_api_key()

    nanochat_base = NANOCHAT_CACHE_ON_VOLUME if args.volume_id else "${HOME}/.cache/nanochat"
    run_name = f"d{args.depth}-{args.ffn_type}"
    smoke = getattr(args, "smoke", False)

    startup = _make_train_startup(
        depth=args.depth,
        ffn_type=args.ffn_type,
        nanochat_base=nanochat_base,
        run_name=run_name,
        smoke=smoke,
        save_every_override=getattr(args, "save_every", None),
        num_iterations=getattr(args, "num_iterations", None),
    )

    if args.dry_run:
        print("Startup script (dry run):")
        print("\n".join(f"  {s}" for s in startup))
        if args.volume_id:
            print(f"\nVolume {args.volume_id} → {VOLUME_MOUNT}")
        return

    # Rough time estimate: d12 at ratio=12 = ~2800 steps × 2.5s/step ≈ 2h on 4090
    est_hours = 2.5 if args.depth <= 12 else 5.0
    if args.ffn_type == "grkan":
        est_hours *= 1.15  # ~15% overhead for rational activations (with CUDA kernel)

    print(f"\nLaunching training pod:")
    print(f"  Depth   : {args.depth}  (n_embd = {args.depth * 64})")
    print(f"  FFN type: {args.ffn_type}")
    print(f"  Run name: {run_name}  (wandb + checkpoint tag)")
    print(f"  Window  : {WINDOW_PATTERN}")
    if args.volume_id:
        print(f"  Volume  : {args.volume_id} → {VOLUME_MOUNT}")
    else:
        print("  Volume  : none (checkpoints only on pod disk — risky on community cloud!)")
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
    print("Pod stops itself when training finishes (disk preserved for download).")
    if args.volume_id:
        print("Checkpoints are on the volume — safe against community-cloud interruption.")
        print(f"If interrupted, relaunch with --resume-from-step <last_saved_step>.")
    print(f"\nAuto-download when done:")
    print(f"  python scripts/runpod_launch.py watch {pod_id}")
    print(f"Or download manually once pod shows Exited:")
    print(f"  python scripts/runpod_launch.py download {pod_id}")


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
    for _ in range(12):
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") or []
        for p in ports:
            if p.get("privatePort") == 22:
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

    print(f"\nCheckpoints saved to: {dest}/")

    try:
        answer = input("Terminate pod (delete disk permanently)? [y/N] ").strip().lower()
    except EOFError:
        answer = "y"
        print("y  (auto-terminated — no stdin)")

    if answer == "y":
        runpod.terminate_pod(args.pod_id)
        print(f"Pod {args.pod_id} terminated.")
    else:
        print(f"Pod kept. Terminate later with:")
        print(f"  python scripts/runpod_launch.py terminate {args.pod_id}")


# ── Subcommand: eval ──────────────────────────────────────────────────────────

# RTX 4090 is sufficient for inference (24 GB, ~$0.34/hr).
EVAL_GPU_PREFERENCE = [
    # Cheap 24–32 GB (prefer first)
    "NVIDIA GeForce RTX 4090",        # 24 GB ~$0.34/hr
    "NVIDIA RTX PRO 4500 Blackwell",  # 32 GB ~$0.34/hr
    "NVIDIA GeForce RTX 3090",        # 24 GB ~$0.24/hr
    "NVIDIA GeForce RTX 3090 Ti",     # 24 GB ~$0.29/hr
    "NVIDIA RTX A5000",               # 24 GB ~$0.30/hr
    "NVIDIA RTX A4500",               # 20 GB ~$0.28/hr
    # Mid-range 48 GB
    "NVIDIA RTX A6000",               # 48 GB ~$0.33/hr
    "NVIDIA A40",                     # 48 GB ~$0.49/hr
    "NVIDIA L40",                     # 48 GB ~$0.69/hr
    "NVIDIA L40S",                    # 48 GB ~$0.79/hr
    "NVIDIA RTX 6000 Ada Generation", # 48 GB ~$0.79/hr
    # 24–32 GB Ada generation
    "NVIDIA L4",                      # 24 GB ~$0.44/hr
    "NVIDIA RTX 5000 Ada Generation", # 32 GB ~$0.49/hr
    "NVIDIA RTX 4000 Ada Generation", # 20 GB ~$0.35/hr
    "NVIDIA RTX PRO 4000 Blackwell",  # 24 GB
    # A100 / H100 (expensive but highly available on secure cloud)
    "NVIDIA A100 80GB PCIe",          # 80 GB ~$1.89/hr
    "NVIDIA A100-SXM4-80GB",          # 80 GB ~$2.29/hr
    "NVIDIA H100 PCIe",               # 80 GB ~$2.49/hr
    "NVIDIA H100 80GB HBM3",          # 80 GB ~$2.99/hr
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

    get_api_key()
    pod_id = args.pod_id
    poll_minutes = args.interval
    dest = Path(args.dest)

    print(f"Watching pod {pod_id}. Will auto-download when training finishes.")
    print(f"Polling every {poll_minutes} min. Keep this terminal open. Ctrl+C to cancel.")

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
        print("\nERROR: SSH port never appeared. Check the pod in the RunPod console.")
        sys.exit(1)

    print(f"\nSSH endpoint: {ssh_ip}:{ssh_port}")

    print("Waiting for sshd ", end="", flush=True)
    for _ in range(30):
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             "-p", str(ssh_port), f"root@{ssh_ip}", "echo ok"],
            capture_output=True,
        )
        if r.returncode == 0:
            print(" ready.\n")
            break
        print(".", end="", flush=True)
        time.sleep(10)
    else:
        print("\nERROR: sshd never became ready.")
        sys.exit(1)

    # Sentinel written by startup script after training finishes.
    # Model tag unknown here (could be d12-mlp or d12-grkan), so check either.
    check_cmd = "if ls ~/nanochat_results/FAILED_* >/dev/null 2>&1; then echo failed; elif ls ~/nanochat_results/DONE_* >/dev/null 2>&1; then echo done; else echo running; fi"
    ssh_opt = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p {ssh_port}"
    ssh_failures = 0
    log_line_offset = 1  # next unread line in the remote training log (1-indexed)

    while True:
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-p", str(ssh_port), f"root@{ssh_ip}", check_cmd],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            ssh_failures += 1
            if ssh_failures >= 3:
                print(f"[{time.strftime('%H:%M')}] SSH failed 3 times — pod may have terminated.")
                sys.exit(1)
            print(f"[{time.strftime('%H:%M')}] SSH check failed ({ssh_failures}/3), retrying in 1 min …")
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

        status = r.stdout.strip().splitlines()[-1]
        if status == "failed":
            print(f"[{time.strftime('%H:%M')}] Training failed on the pod. Starting download of logs/checkpoints …\n")
            class _Args:
                pass

            dl_args = _Args()
            dl_args.pod_id = pod_id
            dl_args.dest = str(dest)
            cmd_download(dl_args)
            sys.exit(1)
        if status == "done":
            print(f"[{time.strftime('%H:%M')}] Training finished! Starting download …\n")

            class _Args:
                pass

            dl_args = _Args()
            dl_args.pod_id = pod_id
            dl_args.dest = str(dest)
            cmd_download(dl_args)
            return

        print(f"[{time.strftime('%H:%M')}] Still training … syncing checkpoints …")
        dest.mkdir(parents=True, exist_ok=True)
        sync = subprocess.run(
            [
                "rsync", "-a", "--ignore-existing", "--partial",
                "--include=*/", "--include=model_*.pt", "--include=meta_*.json",
                "--exclude=*",
                "-e", ssh_opt,
                f"root@{ssh_ip}:~/nanochat_results/base_checkpoints/",
                str(dest) + "/",
            ],
            capture_output=True, text=True,
        )
        new_ckpts = [l.strip() for l in sync.stdout.splitlines() if l.strip().endswith(".pt")]
        if new_ckpts:
            print(f"  Downloaded: {', '.join(new_ckpts)}")
        elif sync.returncode not in (0, 23, 24):
            print(f"  WARNING: rsync exited {sync.returncode}: {sync.stderr.strip()[:120]}")
        print(f"  Next check in {poll_minutes} min.")
        time.sleep(poll_minutes * 60)


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

    p_stop = sub.add_parser("stop", help="Stop a pod (preserves disk)")
    p_stop.add_argument("pod_id")

    p_term = sub.add_parser("terminate", help="Terminate a pod (deletes disk)")
    p_term.add_argument("pod_id")

    p_vol = sub.add_parser("volume", help="Manage persistent network volumes")
    vol_sub = p_vol.add_subparsers(dest="volume_action", required=True)
    p_vc = vol_sub.add_parser("create", help="Create a new volume")
    p_vc.add_argument("--size", type=int, default=10,
                      help="Size in GB (default: 10; covers dataset + tokenizer + checkpoints)")
    vol_sub.add_parser("ls", help="List volumes")
    p_vd = vol_sub.add_parser("delete", help="Delete a volume")
    p_vd.add_argument("volume_id")

    p_test = sub.add_parser("test", help="Run GPU smoke test pod (~5 min, ~$0.03)")
    p_test.add_argument("--gpu", default=None, help="Force a specific GPU type")
    p_test.add_argument("--dry-run", action="store_true",
                        help="Print startup script without launching")

    p_train = sub.add_parser("train", help="Launch a training pod")
    p_train.add_argument("--ffn-type", dest="ffn_type", required=True,
                         choices=["mlp", "grkan"],
                         help="FFN type: mlp (baseline) or grkan (GR-KAN)")
    p_train.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                         help=f"Model depth (default: {DEFAULT_DEPTH})")
    p_train.add_argument("--gpu", default=None, help="Force a specific GPU type")
    p_train.add_argument("--volume-id", default=None,
                         help="Persistent volume ID. Strongly recommended for community cloud.")
    p_train.add_argument("--smoke", action="store_true",
                         help="Smoke-test mode: 10 steps, save-every=10, verify pipeline then stop.")
    p_train.add_argument("--num-iterations", dest="num_iterations", type=int, default=None,
                         help="Override total training steps (default: auto from param-data ratio).")
    p_train.add_argument("--save-every", dest="save_every", type=int, default=None,
                         help="Override save-every (checkpoints per N steps). Overrides smoke default.")
    p_train.add_argument("--secure", action="store_true",
                         help="Use secure cloud (no preemption). Costs more but guaranteed uptime.")
    p_train.add_argument("--dry-run", action="store_true",
                         help="Print startup script without launching")

    p_dl = sub.add_parser("download", help="rsync results from a running/stopped pod")
    p_dl.add_argument("pod_id")
    p_dl.add_argument("--dest", default=LOCAL_DEST,
                      help=f"Local destination directory (default: {LOCAL_DEST})")

    p_watch = sub.add_parser("watch", help="Auto-download when training finishes")
    p_watch.add_argument("pod_id")
    p_watch.add_argument("--dest", default=LOCAL_DEST,
                         help=f"Local destination directory (default: {LOCAL_DEST})")
    p_watch.add_argument("--interval", type=int, default=5,
                         help="Polling interval in minutes (default: 5)")

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
        "download": cmd_download,
        "watch": cmd_watch,
        "eval": cmd_eval,
        "eval-download": cmd_eval_download,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
