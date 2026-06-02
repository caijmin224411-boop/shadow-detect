#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="g_paper_shadow_model")
    args = parser.parse_args()

    data = Path(args.input).read_bytes()
    lines = []
    for i in range(0, len(data), 12):
        chunk = ", ".join(f"0x{b:02x}" for b in data[i : i + 12])
        lines.append(f"  {chunk},")
    content = (
        "#include <cstdint>\n\n"
        f"alignas(16) const unsigned char {args.name}[] = {{\n"
        + "\n".join(lines)
        + "\n};\n"
        f"const unsigned int {args.name}_len = {len(data)};\n"
    )
    Path(args.output).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()

