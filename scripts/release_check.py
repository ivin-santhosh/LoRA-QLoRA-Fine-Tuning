"""Convenience harness: standard-library tests, CLI smoke checks, native CTest."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def check(args: list[str]) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True, timeout=150)


def main() -> int:
    try:
        check([sys.executable, "-m", "compileall", "-q", "modelstack_lab", "scripts", "tests"])
        for path in ("rewards.py", "train_qlora.py", "train_grpo.py"):
            if (ROOT / path).is_file():
                check([sys.executable, "-m", "py_compile", path])
        check([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
        check([sys.executable, "-m", "modelstack_lab", "graph", "--size", "4", "--iterations", "3"])
        if shutil.which("cmake") and (shutil.which("g++") or shutil.which("clang++") or shutil.which("cl")):
            with tempfile.TemporaryDirectory(prefix="modelstack-native-") as tmp:
                check(["cmake", "-S", str(ROOT), "-B", tmp])
                check(["cmake", "--build", tmp, "--config", "Release"])
                check(["ctest", "--test-dir", tmp, "-C", "Release", "--output-on-failure"])
        else:
            print("SKIP native C++ checks: no CMake/C++ compiler detected")
        print("PASS: local release checks (not a GPU or training certification)")
        return 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print("FAIL:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
