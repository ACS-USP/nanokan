"""
GPU environment smoke test for nanochat RunPod pods.

Run this before committing to a long training job to verify:
  - CUDA 12.8 is available (torch 2.9.1 cu128 build, not a CPU-only install)
  - nanochat imports work correctly
  - Forward + backward pass for both mlp and grkan FFN types
  - rational_kat_cu CUDA kernel is loaded (required for grkan training speed)
  - VRAM headroom at d12 scale

Usage (on the pod, from repo root):
    .venv/bin/python scripts/runpod_gpu_test.py

Exit codes:
    0  — all checks passed
    1  — something is broken (error printed to stderr)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def fail(msg: str):
    print(f"\nFAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def check_cuda():
    import torch
    print(f"PyTorch version : {torch.__version__}")
    if not torch.cuda.is_available():
        fail(
            "torch.cuda.is_available() returned False.\n"
            "  Most likely cause: PyTorch was installed from default PyPI (CPU-only build)\n"
            "  or uv installed the wrong extra.\n"
            "  Fix: verify `uv sync --extra gpu` completed without errors and that\n"
            "  torch was installed from https://download.pytorch.org/whl/cu128"
        )
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cuda_version = torch.version.cuda
    print(f"CUDA device     : {name}")
    print(f"VRAM total      : {vram_gb:.1f} GB")
    print(f"CUDA version    : {cuda_version}")
    if cuda_version and tuple(int(x) for x in cuda_version.split(".")[:2]) < (12, 8):
        print(f"  WARNING: torch was built for CUDA {cuda_version}, expected 12.8 (cu128).")
        print("  rational_kat_cu may fail to load. Verify `uv sync --extra gpu`.")
    return device


def check_matmul(device):
    import torch
    print("Matrix multiply … ", end="", flush=True)
    a = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    c = a @ b
    assert c.shape == (1024, 1024)
    torch.cuda.synchronize()
    print("OK")


def check_rational_kat_cu():
    """Verify the fused CUDA kernel is available. This is required for grkan training speed."""
    from nanochat.gpt import _RAT_CUDA_AVAILABLE
    if _RAT_CUDA_AVAILABLE:
        print("rational_kat_cu : LOADED (fused CUDA kernel active)")
    else:
        fail(
            "rational_kat_cu CUDA kernel not loaded.\n"
            "  Without it, GroupRational backward is ~123× slower (pure-PyTorch Horner loop).\n"
            "  This makes grkan training impractical.\n"
            "  Fix: ensure `pip install rational-kat-cu` ran successfully and that nvcc\n"
            "  is available (requires nvidia/cuda:*-devel image, not -runtime).\n"
            "  Check: `nvcc --version` and `pip show rational-kat-cu`"
        )
    return _RAT_CUDA_AVAILABLE


def check_model(device, ffn_type: str):
    import torch
    import torch.nn.functional as F
    from nanochat.gpt import GPT, GPTConfig

    print(f"GPT({ffn_type}) import   … ", end="", flush=True)
    # Small config matching d12 shape (n_embd must be divisible by grkan_groups=8)
    cfg = GPTConfig(
        sequence_len=128,
        vocab_size=512,
        n_layer=2,
        n_head=6,
        n_kv_head=6,
        n_embd=96,          # 96 / 8 groups = 12 per group — valid
        window_pattern="L", # SDPA fallback has no sliding window support
        ffn_type=ffn_type,
    )
    # Build on meta device then materialize — matches nanochat's init pattern exactly
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    model.train()
    print("OK")

    print(f"GPT({ffn_type}) forward  … ", end="", flush=True)
    B, T = 2, 128
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    logits, loss = model(idx, targets)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    assert logits.shape == (B, T, cfg.vocab_size), f"unexpected logits shape {logits.shape}"
    assert not torch.isnan(loss), "loss is NaN"
    print(f"OK  ({elapsed*1000:.0f} ms, loss={loss.item():.3f})")

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"  Peak VRAM     : {peak_mb:.0f} MB")

    if ffn_type == "grkan":
        # Spot-check that rational params receive gradients (after at least one step)
        rat_grads = {
            n: p.grad
            for n, p in model.named_parameters()
            if "rat" in n and p.grad is not None
        }
        print(f"  Rational grads: {len(rat_grads)} params have gradients")
        # Note: c_proj is zero-initialized so rat1.a / rat1.b grads are 0 at init.
        # This is expected — same as standard MLP's c_fc. Not a bug.


def check_throughput(device):
    """Measure throughput for both FFN types at d12 scale."""
    import torch
    from nanochat.gpt import GPT, GPTConfig

    print("\nThroughput check at d12 scale (n_embd=768, seq_len=512):")
    results = {}
    for ffn_type in ("mlp", "grkan"):
        cfg = GPTConfig(
            sequence_len=512, vocab_size=32768,
            n_layer=12, n_head=6, n_kv_head=6, n_embd=768,
            window_pattern="L",
            ffn_type=ffn_type,
        )
        with torch.device("meta"):
            model = GPT(cfg)
        model.to_empty(device=device)
        model.init_weights()
        model.eval()

        idx = torch.randint(0, cfg.vocab_size, (2, 512), device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model(idx)
        torch.cuda.synchronize()

        # Measure
        N = 10
        t0 = time.time()
        with torch.no_grad():
            for _ in range(N):
                model(idx)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        tok_per_sec = N * 2 * 512 / elapsed
        results[ffn_type] = tok_per_sec
        print(f"  {ffn_type:<8}: {tok_per_sec/1000:.1f}k tok/sec")

    ratio = results["mlp"] / results["grkan"]
    print(f"  mlp/grkan ratio: {ratio:.2f}x")
    if ratio > 1.20:
        print(f"  WARNING: grkan is >{ratio:.0%} slower than mlp ({ratio:.2f}×).")
        print("  This may indicate the rational_kat_cu kernel is not being used for inference.")
        print("  For training (with backward), slowdown budget is ~20%.")
    else:
        print(f"  OK: within the ~20% slowdown budget.")


def main():
    print("=" * 55)
    print("nanochat RunPod GPU smoke test")
    print("=" * 55)

    device = check_cuda()
    check_matmul(device)
    rat_ok = check_rational_kat_cu()
    print()

    check_model(device, "mlp")
    print()

    if rat_ok:
        check_model(device, "grkan")
        print()
        check_throughput(device)
    else:
        print("SKIP: grkan model test (rational_kat_cu not available)")

    print()
    print("=" * 55)
    if rat_ok:
        print("ALL CHECKS PASSED — safe to launch training.")
    else:
        print("PARTIAL PASS: mlp OK, grkan blocked (see rational_kat_cu error above).")
        print("Do NOT launch grkan training until rational_kat_cu compiles.")
    print("=" * 55)
    sys.exit(0 if rat_ok else 1)


if __name__ == "__main__":
    main()
