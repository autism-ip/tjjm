"""
 * [INPUT]: 依赖 dataclasses, pathlib, numpy, src.experiments.io
 * [OUTPUT]: 对外提供 summarize_anomaly_maps, summarize_metric_records, save_summary
 * [POS]: src/experiments/ 的分析层, 负责健康样本统计与 ablation/cross-dataset 汇总
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.experiments.io import load_array, load_report
from src.evaluation.reporter import save_report


def _flatten_arrays(arrays: Sequence[np.ndarray]) -> np.ndarray:
    values = [np.asarray(array, dtype=np.float32).reshape(-1) for array in arrays if np.asarray(array).size > 0]
    if not values:
        raise ValueError("arrays cannot be empty")
    return np.concatenate(values)


def summarize_anomaly_maps(
    anomaly_maps: Sequence[np.ndarray],
    percentiles: Sequence[int] = (50, 75, 90, 95),
) -> dict[str, float | int]:
    """
    汇总健康样本的异常图分布统计。
    """
    flat = _flatten_arrays(anomaly_maps)
    summary: dict[str, float | int] = {
        "count": int(len(anomaly_maps)),
        "voxels": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }
    for percentile in percentiles:
        key = f"p{int(percentile)}"
        summary[key] = float(np.percentile(flat, percentile))
    return summary


def summarize_anomaly_map_files(
    paths: Sequence[str | Path],
    percentiles: Sequence[int] = (50, 75, 90, 95),
) -> dict[str, float | int]:
    """对磁盘上的异常图文件做健康分布统计。"""
    maps = [load_array(path) for path in paths]
    return summarize_anomaly_maps(maps, percentiles=percentiles)


def _numeric_fields(records: Sequence[Mapping[str, Any]], group_key: str) -> list[str]:
    fields: set[str] = set()
    for record in records:
        for key, value in record.items():
            if key == group_key:
                continue
            if isinstance(value, (int, float, np.number)):
                fields.add(str(key))
    return sorted(fields)


def summarize_metric_records(
    records: Sequence[Mapping[str, Any]],
    group_key: str = "run_name",
) -> dict[str, dict[str, float | int]]:
    """
    按 run / dataset 聚合实验记录，输出可直接写报告的扁平统计。
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if group_key not in record:
            raise KeyError(f"Missing group key: {group_key}")
        grouped.setdefault(str(record[group_key]), []).append(record)

    summary: dict[str, dict[str, float | int]] = {}
    for group_name, items in grouped.items():
        group_summary: dict[str, float | int] = {"count": int(len(items))}
        for field in _numeric_fields(items, group_key):
            values = np.asarray(
                [
                    float(item[field])
                    for item in items
                    if field in item and isinstance(item[field], (int, float, np.number))
                ],
                dtype=np.float32,
            )
            if values.size == 0:
                continue
            group_summary[f"{field}_mean"] = float(np.mean(values))
            group_summary[f"{field}_std"] = float(np.std(values))
            group_summary[f"{field}_min"] = float(np.min(values))
            group_summary[f"{field}_max"] = float(np.max(values))
        summary[group_name] = group_summary
    return summary


def summarize_metric_reports(
    paths: Sequence[str | Path],
    group_key: str = "run_name",
) -> dict[str, dict[str, float | int]]:
    """读取 JSON 指标报告并做 ablation / cross-dataset 汇总。"""
    records = [load_report(path) for path in paths]
    return summarize_metric_records(records, group_key=group_key)


def compare_metric_records(
    baseline: Mapping[str, Any],
    variant: Mapping[str, Any],
    fields: Sequence[str] | None = None,
) -> dict[str, dict[str, float | None]]:
    """对两个指标记录做逐字段对比，输出 delta 与 delta_pct。"""
    if fields is None:
        shared_fields = set(_numeric_fields([baseline], group_key="__ignored__"))
        shared_fields &= set(_numeric_fields([variant], group_key="__ignored__"))
        fields = sorted(shared_fields)

    comparison: dict[str, dict[str, float | None]] = {}
    for field in fields:
        if field not in baseline or field not in variant:
            continue
        baseline_value = baseline[field]
        variant_value = variant[field]
        if not isinstance(baseline_value, (int, float, np.number)):
            continue
        if not isinstance(variant_value, (int, float, np.number)):
            continue

        base = float(baseline_value)
        value = float(variant_value)
        delta = value - base
        comparison[field] = {
            "baseline": base,
            "variant": value,
            "delta": delta,
            "delta_pct": float(delta / abs(base)) if base != 0.0 else None,
        }
    return comparison


def compare_metric_reports(
    baseline_path: str | Path,
    variant_path: str | Path,
    fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """读取两份 JSON 报告并输出比较结果。"""
    baseline = load_report(baseline_path)
    variant = load_report(variant_path)
    return {
        "baseline_path": str(baseline_path),
        "variant_path": str(variant_path),
        "comparison": compare_metric_records(baseline, variant, fields=fields),
    }


def save_summary(summary: Mapping[str, Any], output_path: str | Path) -> None:
    """将实验摘要写入 JSON。"""
    save_report(dict(summary), output_path)


def save_summary_text(summary: Mapping[str, Any], output_path: str | Path) -> None:
    """保存可直接查看的 JSON 文本，便于竞赛提交后检索。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
