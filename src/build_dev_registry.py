"""Gabungkan sembilan run development menjadi history dan registry pra-test."""
from __future__ import annotations

import argparse
import json

import torch

from common import get_device, load_config, resolve
from train import METHODS, write_development_bundle


def main():
    ap = argparse.ArgumentParser(description="Bekukan registry development")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["_config_path"] = args.config
    results, logs = [], {}
    for method in METHODS:
        aug = cfg["augmentation"]["by_method"][method][
            "head" if method == "ce" else "contrastive"]
        for seed in (42, 43, 44):
            run_id = f"{method}__{aug}__s{seed}"
            run_dir = resolve(cfg["output"]["runs_dir"]) / run_id
            with open(run_dir / "result.json", encoding="utf-8") as f:
                results.append(json.load(f))
            with open(run_dir / "log.json", encoding="utf-8") as f:
                logs[run_id] = json.load(f)
    device = get_device(cfg.get("device", "auto"))
    write_development_bundle(cfg, results, logs, device)


if __name__ == "__main__":
    main()
