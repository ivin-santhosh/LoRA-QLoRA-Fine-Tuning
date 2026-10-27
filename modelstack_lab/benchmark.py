"""Reference-checked benchmark harness for CPU IR and optional PyTorch MLP.

Only report measurements obtained on the active hardware. No hardcoded speedups.
"""
from __future__ import annotations
import math
import platform
import random
import statistics
import time
from typing import Callable
from .ir import Graph, execute, fuse_matmul_bias_relu, demo_graph


def summarize_ns(samples: list[int]) -> dict[str, float]:
    if not samples or any(t <= 0 for t in samples):
        raise ValueError("benchmark must contain positive durations")
    ordered = sorted(samples)
    return {"median_ms": round(statistics.median(ordered) / 1e6, 6),
            "p95_ms": round(ordered[math.ceil(.95 * len(ordered)) - 1] / 1e6, 6),
            "runs": len(samples)}


def timed(fn: Callable[[], object], warmup: int, iterations: int, sync: Callable[[], None] | None = None) -> dict[str, float]:
    if not (1 <= warmup <= 50 and 2 <= iterations <= 1000):
        raise ValueError("warmup must be 1-50 and iterations 2-1000")
    for _ in range(warmup):
        fn()
        if sync:
            sync()
    samples: list[int] = []
    for _ in range(iterations):
        if sync:
            sync()
        start = time.perf_counter_ns()
        fn()
        if sync:
            sync()
        samples.append(max(1, time.perf_counter_ns() - start))
    return summarize_ns(samples)


def graph_benchmark(size: int = 12, iterations: int = 15, seed: int = 42) -> dict[str, object]:
    if not 1 <= size <= 96:
        raise ValueError("CPU reference graph size must be in [1, 96]")
    randomizer = random.Random(seed)
    feeds = {"x": [[randomizer.uniform(-1, 1) for _ in range(size)] for _ in range(size)],
             "w": [[randomizer.uniform(-1, 1) for _ in range(size)] for _ in range(size)],
             "b": [randomizer.uniform(-1, 1) for _ in range(size)]}
    graph = demo_graph()
    fused = fuse_matmul_bias_relu(graph)
    reference, reference_meta = execute(graph, feeds)
    result, fused_meta = execute(fused, feeds)
    max_error = max(abs(a - b) for ar, br in zip(reference, result) for a, b in zip(ar, br))
    if max_error > 1e-9:
        raise AssertionError("fusion correctness check failed")
    return {"tool": "reference-tensor-ir", "backend": "python-cpu", "shape": [size, size],
            "seed": seed, "verified_max_abs_error": max_error,
            "eager": {**timed(lambda: execute(graph, feeds), 2, iterations), **reference_meta},
            "fused": {**timed(lambda: execute(fused, feeds), 2, iterations), **fused_meta},
            "meaning": "Timing is environment-dependent; intermediate_output_elements is a logical materialization count, not peak RAM."}


def torch_benchmark(size: int = 128, iterations: int = 15, seed: int = 42,
                    device: str = "cpu", use_compile: bool = False) -> dict[str, object]:
    if size < 8 or size > 1024:
        raise ValueError("torch size must be in [8, 1024]")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch unavailable: install torch separately for this optional benchmark") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu")
    torch.manual_seed(seed)
    class TinyMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(size, size)
            self.fc2 = torch.nn.Linear(size, size)
        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))
    model = TinyMLP().eval().to(device)
    inputs = torch.randn((8, size), device=device)
    sync = torch.cuda.synchronize if device == "cuda" else None
    with torch.inference_mode():
        ref = model(inputs)
        eager = timed(lambda: model(inputs), 2, iterations, sync)
        result: dict[str, object] = {"tool": "pytorch-mlp", "torch_version": torch.__version__,
                                    "device": device, "size": size, "batch": 8, "seed": seed,
                                    "eager": eager, "compile_requested": use_compile}
        if use_compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is unavailable in this PyTorch installation")
            candidate = torch.compile(model)
            # Compilation and warmup are deliberately excluded from steady-state timing.
            got = candidate(inputs)
            if sync:
                sync()
            error = (ref - got).abs().max().item()
            tolerance = 1e-4 + 1e-4 * ref.abs().max().item()
            if error > tolerance or not math.isfinite(error):
                raise AssertionError("compiled result differs from eager reference")
            result["compiled"] = timed(lambda: candidate(inputs), 2, iterations, sync)
            result["compiled_max_abs_error"] = error
        result["note"] = "Synthetic inference microbenchmark; not training or production throughput."
        return result
