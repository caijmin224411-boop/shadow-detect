#!/usr/bin/env python3
from __future__ import annotations

import py_compile
from pathlib import Path


def main():
    roots = [Path("src"), Path("scripts")]
    files = []
    for root in roots:
        files.extend(root.rglob("*.py"))
    for path in files:
        cache_path = Path("work/py_compile") / path.with_suffix(".pyc")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        py_compile.compile(str(path), cfile=str(cache_path), doraise=True)
    required = [
        Path("README.md"),
        Path("requirements.txt"),
        Path("esp32/README.md"),
        Path("esp32/shadowcam_runtime.cpp"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    print(f"OK: compiled {len(files)} Python files")


if __name__ == "__main__":
    main()
