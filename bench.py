"""Generic benchmarking / profiling helpers for attention implementations.

Every function here takes an `attn_fn` with the signature:

    attn_fn(q, k, v) -> out        # q,k,v: (B, H, N, D)

so the same tooling works for the naive CUDA kernels, a Triton kernel,
PyTorch SDPA, or anything else you write later.

Usage in Colab:

    import bench
    bench.report(mod.naive_attention, name="naive cuda")
"""

import gc
import math
from contextlib import contextmanager

import torch
from torch.profiler import ProfilerActivity, profile

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_qkv(b=2, h=4, n=1024, d=64, dtype=torch.float32, device="cuda", seed=0):
    """Deterministic (B, H, N, D) inputs."""
    g = torch.Generator(device=device).manual_seed(seed)
    return [
        torch.randn(b, h, n, d, device=device, dtype=dtype, generator=g)
        for _ in range(3)
    ]


def sdpa(q, k, v):
    """PyTorch reference, for passing as an attn_fn."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def reference(q, k, v):
    """fp64 ground truth -- slow, use only for correctness checks."""
    out = torch.nn.functional.scaled_dot_product_attention(
        q.double(), k.double(), v.double()
    )
    return out.to(q.dtype)


@contextmanager
def _clean():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def _device_time(evt):
    """PyTorch renamed cuda_time_total -> device_time_total; support both."""
    for attr in ("device_time_total", "cuda_time_total"):
        if hasattr(evt, attr):
            return getattr(evt, attr)
    return 0.0


# --------------------------------------------------------------------------
# 1. correctness
# --------------------------------------------------------------------------


def check(attn_fn, b=2, h=4, n=512, d=64, rtol=1e-4, atol=1e-5, verbose=True):
    """Compare attn_fn against an fp64 reference. Returns (passed, max_abs_err)."""
    q, k, v = make_qkv(b, h, n, d)
    ref = reference(q, k, v)
    out = attn_fn(q, k, v)

    bad = torch.isnan(out).any().item() or torch.isinf(out).any().item()
    err = (out - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    ok = (not bad) and torch.allclose(out, ref, rtol=rtol, atol=atol)

    if verbose:
        flag = "PASS" if ok else "FAIL"
        extra = "  <- NaN/Inf present" if bad else ""
        print(f"[{flag}] B={b} H={h} N={n} D={d}   "
              f"max_abs={err:.3e}  rel={rel:.3e}{extra}")
    return ok, err


# --------------------------------------------------------------------------
# 2. timing + memory
# --------------------------------------------------------------------------


def time_fn(attn_fn, q, k, v, warmup=5, iters=20):
    """Median-of-iters wall time in ms, plus peak memory in MB."""
    for _ in range(warmup):
        attn_fn(q, k, v)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        attn_fn(q, k, v)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2], torch.cuda.max_memory_allocated() / 2**20


def sweep(attn_fn, ns=(512, 1024, 2048, 4096, 8192), b=2, h=4, d=64,
          warmup=3, iters=10, verbose=True):
    """Time + memory across sequence lengths. Stops cleanly on OOM.

    Returns a list of dicts: n, ms, peak_mb, sp_mb, tflops, ai
    """
    rows = []
    if verbose:
        print(f"{'N':>6} {'ms':>9} {'TFLOP/s':>9} {'peak MB':>10} "
              f"{'S+P MB':>9} {'AI':>7}")

    for n in ns:
        try:
            with _clean():
                q, k, v = make_qkv(b, h, n, d)
                ms, peak = time_fn(attn_fn, q, k, v, warmup, iters)

                flops = 4 * b * h * n * n * d          # two matmuls
                nsq_bytes = 4 * b * h * n * n * 4      # S/P write+read, x2, fp32
                row = dict(
                    n=n, ms=ms, peak_mb=peak,
                    sp_mb=b * h * n * n * 4 / 2**20,
                    tflops=flops / (ms * 1e-3) / 1e12,
                    ai=flops / nsq_bytes,
                )
                rows.append(row)
                if verbose:
                    print(f"{n:>6} {ms:>9.3f} {row['tflops']:>9.2f} "
                          f"{peak:>10.1f} {row['sp_mb']:>9.1f} {row['ai']:>7.1f}")
                del q, k, v
        except torch.cuda.OutOfMemoryError:
            if verbose:
                print(f"{n:>6} {'OOM':>9}   <-- quadratic memory wall")
            gc.collect()
            torch.cuda.empty_cache()
            break
    return rows


def compare(fns, ns=(512, 1024, 2048, 4096), b=2, h=4, d=64, **kw):
    """Benchmark several implementations side by side.

    fns: dict of {name: attn_fn}
    """
    results = {}
    for name, fn in fns.items():
        print(f"\n=== {name} ===")
        results[name] = sweep(fn, ns, b, h, d, verbose=True, **kw)

    names = list(fns)
    base = names[0]
    print(f"\n{'N':>6} " + " ".join(f"{n[:12]:>13}" for n in names)
          + f"   (speedup vs {base})")
    by_n = {name: {r['n']: r['ms'] for r in rows} for name, rows in results.items()}
    for n in ns:
        if not any(n in by_n[name] for name in names):
            continue
        cells = []
        for name in names:
            ms = by_n[name].get(n)
            if ms is None:
                cells.append(f"{'OOM':>13}")
            else:
                sp = by_n[base].get(n)
                mark = f" ({sp/ms:.1f}x)" if sp and name != base else ""
                cells.append(f"{ms:>8.2f}ms{mark:>5}")
        print(f"{n:>6} " + " ".join(cells))
    return results


# --------------------------------------------------------------------------
# 3. per-kernel profile
# --------------------------------------------------------------------------


def kernels(attn_fn, b=2, h=4, n=2048, d=64, iters=10, top=15, verbose=True):
    """Per-CUDA-kernel time breakdown. Returns list of (name, ms_per_iter, pct)."""
    with _clean():
        q, k, v = make_qkv(b, h, n, d)
        for _ in range(3):
            attn_fn(q, k, v)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                attn_fn(q, k, v)
            torch.cuda.synchronize()

        evts = [
            e for e in prof.key_averages()
            if _device_time(e) > 0 and "memcpy" not in e.key.lower()
        ]
        total = sum(_device_time(e) for e in evts) or 1.0

        rows = [
            (e.key, _device_time(e) / 1000 / iters, 100 * _device_time(e) / total)
            for e in sorted(evts, key=lambda x: -_device_time(x))
        ]
        if verbose:
            print(f"{'kernel':<44}{'ms/iter':>10}{'%':>8}")
            for name, ms, pct in rows[:top]:
                print(f"{name[:43]:<44}{ms:>10.3f}{pct:>7.1f}%")
        del q, k, v
    return rows


# --------------------------------------------------------------------------
# 4. everything at once
# --------------------------------------------------------------------------


def report(attn_fn, name="impl", ns=(512, 1024, 2048, 4096, 8192),
           b=2, h=4, d=64, profile_n=2048, check_n=512):
    """Correctness + sweep + kernel breakdown + roofline context."""
    p = torch.cuda.get_device_properties(0)
    print(f"### {name}  on {p.name} (sm_{p.major}{p.minor}, "
          f"{p.multi_processor_count} SMs, {p.total_memory/2**30:.1f} GB)\n")

    print("-- correctness --")
    ok, _ = check(attn_fn, b, h, check_n, d)

    print("\n-- scaling (B=%d H=%d D=%d) --" % (b, h, d))
    rows = sweep(attn_fn, ns, b, h, d)

    print(f"\n-- kernel breakdown (N={profile_n}) --")
    kernels(attn_fn, b, h, profile_n, d)

    if rows:
        print("\n-- roofline --")
        r = rows[-1]
        print(f"arithmetic intensity at N={r['n']}: {r['ai']:.1f} FLOP/byte "
              f"(counting only N^2 traffic)")
        print("compare against this GPU's ops:byte ratio -- if AI is far below "
              "it, you are memory bound and fusion is the fix.")
    return dict(ok=ok, rows=rows)


def plot(results, metric="ms"):
    """Optional matplotlib plot. results = {name: rows} from compare(), or rows."""
    import matplotlib.pyplot as plt

    if isinstance(results, list):
        results = {"impl": results}
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, rows in results.items():
        ax.plot([r["n"] for r in rows], [r[metric] for r in rows],
                marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length N")
    ax.set_ylabel({"ms": "time (ms)", "peak_mb": "peak memory (MB)",
                   "tflops": "TFLOP/s"}.get(metric, metric))
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    return fig
