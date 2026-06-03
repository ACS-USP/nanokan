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
import torch
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


def _canonical_reference(x, a, b, groups):
    """Canonical Safe Padé reference: P(x)/(1 + |sum_i b_i*x^(i+1)|)."""
    x_g = x.reshape(-1, groups, x.shape[-1] // groups)
    num = a[-1]
    for i in range(a.numel() - 2, -1, -1):
        num = a[i] + x_g * num
    denom_poly = torch.zeros_like(x_g)
    x_power = x_g
    for i in range(b.shape[1]):
        denom_poly = denom_poly + b[:, i].view(1, groups, 1) * x_power
        x_power = x_power * x_g
    return (num / (1.0 + denom_poly.abs())).reshape_as(x)


def _wrong_reference(x, a, b, groups):
    """Retracted formula: P(x)/(1 + sum_i |b_i|*|x|^(i+1))."""
    x_g = x.reshape(-1, groups, x.shape[-1] // groups)
    num = a[-1]
    for i in range(a.numel() - 2, -1, -1):
        num = a[i] + x_g * num
    denom = torch.ones_like(x_g)
    x_abs_power = x_g.abs()
    for i in range(b.shape[1]):
        denom = denom + b[:, i].abs().view(1, groups, 1) * x_abs_power
        x_abs_power = x_abs_power * x_g.abs()
    return (num / denom).reshape_as(x)


def check_grkan_formula_kernel(device):
    import torch
    from nanochat.gpt import GroupRational

    print("GR-KAN formula gate  … ", end="", flush=True)
    groups = 2
    layer = GroupRational(d_in=8, num_groups=groups, m=5, n=4, init="identity").to(device)
    with torch.no_grad():
        layer.a.copy_(torch.tensor([0.2, -0.4, 0.9, -0.2, 0.05, 0.01], device=device))
        layer.b.copy_(torch.tensor([[0.9, -0.7, 0.25, -0.1], [-0.6, 0.8, -0.35, 0.15]], device=device))

    x = torch.tensor(
        [[[-2.0, -0.75, 0.25, 1.5, 1.25, -1.1, 0.6, -0.3],
          [0.9, -1.4, 1.8, -0.2, -1.7, 0.4, -0.8, 1.1]]],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    weight = torch.linspace(-0.7, 0.9, x.numel(), device=device, dtype=torch.float32).reshape_as(x)

    got = layer(x)
    expected = _canonical_reference(x, layer.a, layer.b, groups)
    wrong = _wrong_reference(x, layer.a, layer.b, groups)
    max_abs = (got - expected).abs().max().item()
    wrong_abs = (got - wrong).abs().max().item()
    if max_abs > 2e-4:
        fail(f"fused GR-KAN forward does not match canonical formula (max_abs={max_abs:.6g})")
    if wrong_abs < 1e-3:
        fail(f"adversarial case did not separate wrong formula (max_abs={wrong_abs:.6g})")

    fused_loss = (got * weight).sum()
    fused_grads = torch.autograd.grad(fused_loss, (x, layer.a, layer.b), retain_graph=False)

    x_ref = x.detach().clone().requires_grad_(True)
    a_ref = layer.a.detach().clone().requires_grad_(True)
    b_ref = layer.b.detach().clone().requires_grad_(True)
    ref = _canonical_reference(x_ref, a_ref, b_ref, groups)
    ref_loss = (ref * weight).sum()
    ref_grads = torch.autograd.grad(ref_loss, (x_ref, a_ref, b_ref), retain_graph=False)
    grad_names = ("x", "a", "b")
    for name, got_grad, ref_grad in zip(grad_names, fused_grads, ref_grads):
        grad_abs = (got_grad - ref_grad).abs().max().item()
        if grad_abs > 5e-4:
            fail(f"fused GR-KAN {name}-gradient mismatch (max_abs={grad_abs:.6g})")
    print(f"OK  (forward {max_abs:.2e}, wrong-formula sep {wrong_abs:.2e})")


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
    loss = model(idx, targets)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    with torch.no_grad():
        logits = model(idx)
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
        check_grkan_formula_kernel(device)
        print()
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
