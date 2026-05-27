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
    print("ERROR: runpod SDK not installed.  Run: pip install runpod")
    sys.exit(1)


# ── Configuration ─────────────────────────────────────────────────────────────
REPO_URL = "https://github.com/HCAI-USP/nanokan"
BRANCH = "master"

# CUDA 12.8 devel image: required for nvcc to compile rational_kat_cu.
# Do NOT use -runtime — it lacks nvcc and rational_kat_cu will silently fall
# back to the pure-PyTorch Horner loop, which is ~123× slower on backward.
DOCKER_IMAGE = "nvidia/cuda:12.9.2-cudnn-devel-ubuntu22.04"

# GPU preference: cheapest 24 GB+ first. All support CUDA 12.8 (driver >= 570.xx).
GPU_PREFERENCE = [
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
SAVE_EVERY = 250

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
MACHINE_BLACKLIST: set[str] = set()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        print("ERROR: RUNPOD_API_KEY not set.  Run: export RUNPOD_API_KEY=<your-key>")
        sys.exit(1)
    runpod.api_key = key
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
        cloud_type="COMMUNITY",
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
) -> tuple[str, dict]:
    # If a specific GPU is requested, only try that one — do not fall through to others.
    candidates = [preferred_gpu] if preferred_gpu else GPU_PREFERENCE
    print("Finding available GPU …")
    for gpu in candidates:
        try:
            pod = _try_launch_pod(name, gpu, startup, volume_id, disk_gb, env_extra)
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
) -> list[str]:
    cmds = _base_startup()

    # NOTE: rational_kat_cu CUDA kernel is NOT currently installable.
    # The Adamdad/rational_kat_cu repo (kat-rational==0.4) has ext_modules commented
    # out in setup.py, so no CUDA extension compiles and `rational_kat_cu` is never
    # importable. nanochat/gpt.py handles this gracefully: _RAT_CUDA_AVAILABLE=False
    # and GroupRational uses the pure-PyTorch Horner fallback instead.
    # Training is correct but ~2-5x slower for grkan FFN backward passes.
    # TODO: fork the repo, uncomment ext_modules with name='rational_kat_cu', and fix
    #       gpt.py to use rational_fwd_1dgroup / rational_bwd_1dgroup via autograd.Function.

    cmds += _dataset_and_tokenizer_startup(nanochat_base)

    model_tag = f"d{depth}-{ffn_type}"
    log_file = f"{nanochat_base}/{model_tag}_train.log"

    # core-metric-every=-1 disables the expensive DCLM CORE eval during training.
    # Run base_eval.py once after both runs are done to get the final CORE score.
    # Use a subshell with pipefail so tee doesn't mask torchrun's exit code.
    # Without this, `torchrun ... | tee` always returns 0 (tee succeeds) and the
    # && chain incorrectly continues to touch DONE even when training failed.
    # grkan's pure-PyTorch Horner loop keeps more activation intermediates than MLP,
    # using ~42 GB on L40S at the default device_batch_size=32 before the logits
    # allocation (65536 × 32768 × fp32 = 8 GB). Halving to 16 halves both activations
    # and the logits tensor, fitting within 44 GB. grad_accum doubles to compensate.
    device_batch_size = 16 if ffn_type == "grkan" else 32

    train_cmd = (
        f"( set -o pipefail;"
        f" .venv/bin/torchrun --standalone --nproc_per_node=1 -m scripts.base_train --"
        f" --depth={depth}"
        f" --ffn-type={ffn_type}"
        f" --model-tag={model_tag}"
        f" --run={run_name}"
        f" --window-pattern={WINDOW_PATTERN}"
        f" --save-every={SAVE_EVERY}"
        f" --core-metric-every=-1"
        f" --device-batch-size={device_batch_size}"
        f" 2>&1 | tee {log_file})"
    )
    # On failure: sleep infinity so the pod stays alive for SSH debugging.
    # exit 1 (or pipefail) would cause RunPod to restart the container in a loop.
    # The log at {log_file} contains the full Python traceback before the torchrun
    # ChildFailedError — check it first when debugging.
    cmds.append(
        f"{train_cmd}"
        f" || {{ echo TRAINING FAILED — pod sleeping for debug. Check log: {log_file}; sleep infinity; }}"
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
        "{ uv pip install rational-kat-cu --quiet"
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

    startup = _make_train_startup(
        depth=args.depth,
        ffn_type=args.ffn_type,
        nanochat_base=nanochat_base,
        run_name=run_name,
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

    gpu, pod = find_and_launch_pod(
        name=f"nanochat-{run_name}",
        startup=startup,
        preferred_gpu=args.gpu,
        volume_id=args.volume_id,
        disk_gb=60,
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
        "rsync", "-avz", "--progress", "-e", ssh_opt,
        f"{remote}:~/nanochat_results/base_checkpoints/", str(dest) + "/",
    ]
    print(f"\nRunning: {' '.join(results_cmd)}")
    result = subprocess.run(results_cmd)
    if result.returncode != 0:
        print(f"\nERROR: rsync of base_checkpoints/ failed (exit {result.returncode}). Pod NOT terminated.")
        print(f"Try manually: ssh -p {ssh_port} root@{ssh_ip}")
        sys.exit(1)

    # Also grab training logs (exit 23 = no matching files — not an error)
    logs_cmd = [
        "rsync", "-avz", "--progress", "-e", ssh_opt,
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
    check_cmd = "ls ~/nanochat_results/DONE_* 2>/dev/null && echo done || echo running"
    ssh_failures = 0

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
        status = r.stdout.strip().splitlines()[-1]
        if status == "done":
            print(f"[{time.strftime('%H:%M')}] Training finished! Starting download …\n")

            class _Args:
                pass

            dl_args = _Args()
            dl_args.pod_id = pod_id
            dl_args.dest = str(dest)
            cmd_download(dl_args)
            return

        print(f"[{time.strftime('%H:%M')}] Still training … next check in {poll_minutes} min.")
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
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
