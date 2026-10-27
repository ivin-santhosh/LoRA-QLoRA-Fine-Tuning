"""No-download, resource-bounded command-line interface."""
from __future__ import annotations
import argparse
import json
import platform
import shutil
import sys
from .benchmark import graph_benchmark, torch_benchmark
from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m modelstack_lab", description="ModelStack Lab reproducible ML experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect local tool availability; no GPU workload is launched")
    graph = commands.add_parser("graph", help="Run guarded tensor-graph optimization and CPU measurements")
    graph.add_argument("--size", type=int, default=12)
    graph.add_argument("--iterations", type=int, default=15)
    graph.add_argument("--seed", type=int, default=42)
    bench = commands.add_parser("torch-bench", help="Optional PyTorch synthetic MLP inference benchmark")
    bench.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    bench.add_argument("--size", type=int, default=128)
    bench.add_argument("--iterations", type=int, default=15)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--compile", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result: dict[str, object] = {"version": __version__, "python": platform.python_version(),
                "os": platform.platform(), "cmake_available": bool(shutil.which("cmake")),
                "cxx_available": bool(shutil.which("g++") or shutil.which("clang++") or shutil.which("cl")),
                "torch_installed": False, "cuda_available": False}
            try:
                import torch
                result["torch_installed"] = True
                result["torch_version"] = torch.__version__
                result["cuda_available"] = torch.cuda.is_available()
            except ImportError:
                pass
        elif args.command == "graph":
            result = graph_benchmark(size=args.size, iterations=args.iterations, seed=args.seed)
        else:
            result = torch_benchmark(size=args.size, iterations=args.iterations, seed=args.seed,
                                     device=args.device, use_compile=args.compile)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (AssertionError, RuntimeError, ValueError, OverflowError) as exc:
        print(f"modelstack_lab: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
