#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 argparse、yaml、src.experiments 的公共接口
 * [OUTPUT]: 对外提供健康统计、合成异常敏感性、实验协议与 ablation 汇总的 main() 函数
 * [POS]: scripts/ 的实验入口，被 CLI 直接调用
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments import (
    build_experiment_protocol,
    compare_metric_reports,
    evaluate_synthetic_sensitivity,
    iter_input_paths,
    load_array,
    make_difference_score_fn,
    save_experiment_protocol,
    save_summary_text,
    summarize_anomaly_map_files,
    summarize_metric_reports,
)
from src.evaluation import evaluate_luna16_detection_dir

DEFAULT_CONFIG_PATH = Path(PROJECT_ROOT) / "configs" / "experiments.yaml"


def load_experiment_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """读取 experiments.yaml，缺省时返回空配置。"""
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 experiments CLI。"""
    parser = argparse.ArgumentParser(
        description="Run health analysis, weak LUNA16 evaluation, synthetic sensitivity, experiment planning, and ablation summaries.",
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

    compare = subparsers.add_parser("compare", help="Compare one baseline report against one variant report.")
    compare.add_argument("--baseline", type=str, required=True, help="Baseline JSON metrics report.")
    compare.add_argument("--variant", type=str, required=True, help="Variant JSON metrics report.")
    compare.add_argument("--fields", nargs="*", default=None, help="Optional subset of numeric fields to compare.")
    compare.add_argument("--output", type=str, default=None, help="Output JSON path.")

    luna16 = subparsers.add_parser("luna16", help="Evaluate anomaly maps against LUNA16 weak labels.")
    luna16.add_argument("--input-dir", type=str, required=True, help="Directory containing anomaly map files.")
    luna16.add_argument("--annotations", type=str, default=None, help="Path to LUNA16 annotations.csv.")
    luna16.add_argument("--ct-dir", type=str, required=True, help="Directory containing reference LUNA16 .mhd files.")
    luna16.add_argument("--output", type=str, default=None, help="Output JSON path.")
    luna16.add_argument("--score-percentile", type=float, default=None, help="Global anomaly percentile used as positive threshold.")
    luna16.add_argument("--score-percentiles", nargs="*", type=float, default=None, help="Optional percentile sweep for FROC-like operating points.")
    luna16.add_argument("--min-diameter-mm", type=float, default=None, help="Ignore nodules smaller than this diameter.")

    plan = subparsers.add_parser("plan", help="Render the full experiment protocol as Markdown or JSON.")
    plan.add_argument("--output", type=str, default=None, help="Output file path.")
    plan.add_argument("--format", choices=["md", "json"], default="md", help="Output format.")

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


def _run_compare(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    output_dir = config.get("experiments", {}).get("output_dir", "./outputs/experiments")
    summary = compare_metric_reports(args.baseline, args.variant, fields=args.fields)
    output_path = _resolve_output(output_dir, "compare_summary.json", args.output)
    save_summary_text(summary, output_path)
    return summary


def _run_luna16(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    defaults = config.get("experiments", {}).get("luna16", {})
    output_dir = config.get("experiments", {}).get("output_dir", "./outputs/experiments")
    annotations = args.annotations or defaults.get("annotations") or str(Path(args.ct_dir) / "annotations.csv")
    score_percentile = (
        args.score_percentile if args.score_percentile is not None else defaults.get("score_percentile", 99.5)
    )
    score_percentiles = args.score_percentiles if args.score_percentiles else defaults.get(
        "score_percentiles",
        [99.0, 99.5, 99.9],
    )
    min_diameter_mm = (
        args.min_diameter_mm if args.min_diameter_mm is not None else defaults.get("min_diameter_mm", 0.0)
    )
    input_paths = iter_input_paths(input_dir=args.input_dir)
    if not input_paths:
        raise ValueError("luna16 requires anomaly map files under --input-dir")

    summary = evaluate_luna16_detection_dir(
        anomaly_paths=input_paths,
        annotations_path=annotations,
        ct_dir=args.ct_dir,
        score_percentile=score_percentile,
        min_diameter_mm=min_diameter_mm,
        score_percentiles=score_percentiles,
    )
    output_path = _resolve_output(output_dir, "luna16_weak_eval.json", args.output)
    save_summary_text(summary, output_path)
    return summary


def _run_plan(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    output_dir = config.get("experiments", {}).get("output_dir", "./outputs/experiments")
    protocol = build_experiment_protocol(config)
    default_name = "experiment_plan.json" if args.format == "json" else "experiment_plan.md"
    output_path = _resolve_output(output_dir, default_name, args.output)
    save_experiment_protocol(protocol, output_path, format=args.format)
    return protocol


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """运行 experiments CLI。"""
    args = parse_args(argv)
    config = load_experiment_config(args.config)

    if args.command == "health":
        return _run_health(args, config)
    if args.command == "synthetic":
        return _run_synthetic(args, config)
    if args.command == "ablation":
        return _run_ablation(args, config)
    if args.command == "compare":
        return _run_compare(args, config)
    if args.command == "luna16":
        return _run_luna16(args, config)
    if args.command == "plan":
        return _run_plan(args, config)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
