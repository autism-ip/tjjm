#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 argparse, yaml, src.experiments 的实验分析函数
 * [OUTPUT]: 对外提供健康统计、合成异常和 ablation 汇总入口 main()
 * [POS]: scripts/ 的实验入口，面向竞赛交付的离线分析流程
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments import (
    evaluate_synthetic_sensitivity,
    iter_input_paths,
    load_array,
    load_report,
    make_difference_score_fn,
    save_summary_text,
    summarize_anomaly_map_files,
    summarize_metric_reports,
)

DEFAULT_CONFIG_PATH = Path(PROJECT_ROOT) / "configs" / "experiments.yaml"


def load_experiment_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """读取实验默认配置。"""
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析实验 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description="Run compact competition experiments for health analysis, synthetic sensitivity, and ablation summaries.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to experiments.yaml with default outputs and parameter presets.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    health = subparsers.add_parser("health", help="Summarize anomaly statistics for healthy cases.")
    health.add_argument("--inputs", nargs="*", default=[], help="Input anomaly map files (.npy/.npz/.nii.gz/.mhd).")
    health.add_argument("--input-dir", type=str, default=None, help="Directory containing anomaly map files.")
    health.add_argument("--output", type=str, default=None, help="Output JSON path.")
    health.add_argument("--percentiles", nargs="*", type=int, default=None, help="Percentiles to compute.")

    synthetic = subparsers.add_parser("synthetic", help="Evaluate synthetic anomaly sensitivity.")
    synthetic.add_argument("--volume", type=str, required=True, help="Reference 3D volume file.")
    synthetic.add_argument("--output", type=str, default=None, help="Output JSON path.")
    synthetic.add_argument("--radii", nargs="*", type=int, default=None, help="Sphere radii.")
    synthetic.add_argument("--intensities", nargs="*", type=float, default=None, help="Injection intensities.")
    synthetic.add_argument("--center", nargs=3, type=int, default=None, metavar=("Z", "Y", "X"))
    synthetic.add_argument("--mode", choices=["add", "set"], default="add", help="Injection mode.")

    ablation = subparsers.add_parser("ablation", help="Aggregate run reports for ablation or cross-dataset comparison.")
    ablation.add_argument("--reports", nargs="+", required=True, help="JSON metrics reports to compare.")
    ablation.add_argument("--group-key", type=str, default="run_name", help="Field used to group reports.")
    ablation.add_argument("--output", type=str, default=None, help="Output JSON path.")

    return parser.parse_args(argv)


def _resolve_output(default_dir: str | Path, filename: str, explicit: str | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    return Path(default_dir) / filename


def _run_health(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    defaults = config.get("experiments", {}).get("health", {})
    percentiles = args.percentiles if args.percentiles else defaults.get("percentiles", [50, 75, 90, 95])
    output_dir = config.get("experiments", {}).get("output_dir", "./outputs/experiments")
    input_paths = iter_input_paths(args.inputs, args.input_dir)
    if not input_paths:
        raise ValueError("health requires at least one anomaly map input")

    summary = summarize_anomaly_map_files(input_paths, percentiles=percentiles)
    output_path = _resolve_output(output_dir, "health_summary.json", args.output)
    save_summary_text(summary, output_path)
    return summary


def _run_synthetic(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    defaults = config.get("experiments", {}).get("synthetic", {})
    radii = args.radii if args.radii else defaults.get("radii", [4, 6, 8])
    intensities = args.intensities if args.intensities else defaults.get("intensities", [0.5, 1.0])
    output_dir = config.get("experiments", {}).get("output_dir", "./outputs/experiments")

    volume = load_array(args.volume)
    score_fn = make_difference_score_fn(volume)
    results = evaluate_synthetic_sensitivity(
        volume,
        score_fn,
        radii=radii,
        intensities=intensities,
        center=args.center,
        mode=args.mode,
    )

    output_path = _resolve_output(output_dir, "synthetic_summary.json", args.output)
    save_summary_text({"results": results}, output_path)
    return {"results": results}


def _run_ablation(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    output_dir = config.get("experiments", {}).get("output_dir", "./outputs/experiments")
    summary = summarize_metric_reports(args.reports, group_key=args.group_key)
    output_path = _resolve_output(output_dir, "ablation_summary.json", args.output)
    save_summary_text(summary, output_path)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """实验入口主函数。"""
    args = parse_args(argv)
    config = load_experiment_config(args.config)

    if args.command == "health":
        return _run_health(args, config)
    if args.command == "synthetic":
        return _run_synthetic(args, config)
    if args.command == "ablation":
        return _run_ablation(args, config)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
