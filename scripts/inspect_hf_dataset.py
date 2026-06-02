#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path


class TimeoutError(RuntimeError):
    pass


def timeout_handler(_signum, _frame):
    raise TimeoutError("Timed out while contacting Hugging Face")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Donghyun99/ISTD")
    parser.add_argument("--cache-dir", default="work/hf_cache")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)
    try:
        from datasets import get_dataset_config_names, get_dataset_split_names
        from huggingface_hub import HfApi

        api = HfApi()
        files = api.list_repo_files(args.dataset, repo_type="dataset")
        configs = get_dataset_config_names(args.dataset, cache_dir=str(cache_dir / "datasets"))
        split_map = {}
        for config in configs:
            split_map[config] = get_dataset_split_names(
                args.dataset,
                config_name=config,
                cache_dir=str(cache_dir / "datasets"),
            )
        print(json.dumps({"dataset": args.dataset, "ok": True, "files": files, "configs": configs, "splits": split_map}, indent=2))
    except Exception as exc:
        print(json.dumps({"dataset": args.dataset, "ok": False, "error": repr(exc)}, indent=2))
        raise SystemExit(1)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()

