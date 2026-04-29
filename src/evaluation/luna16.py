"""
 * [INPUT]: 依赖 csv, pathlib, numpy, SimpleITK, sklearn.metrics, skimage.measure, src.data.patches
 * [OUTPUT]: 对外提供 load_luna16_annotations, evaluate_luna16_case, evaluate_luna16_detection_dir, evaluate_luna16_threshold_sweep
 * [POS]: src/evaluation/ 的 LUNA16 弱标注评估器，把结节中心直径标注映射到异常图并输出病例级、结节级与阈值扫描指标
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import SimpleITK as sitk
from sklearn.metrics import average_precision_score, roc_auc_score
from skimage.measure import label

from src.data.patches import world_to_voxel


def _load_array(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    suffixes = "".join(file_path.suffixes).lower()

    if suffixes.endswith(".npy"):
        return np.asarray(np.load(file_path))
    if suffixes.endswith(".npz"):
        archive = np.load(file_path)
        key = "arr_0" if "arr_0" in archive else next(iter(archive.files))
        return np.asarray(archive[key])
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz") or suffixes.endswith(".mhd"):
        image = sitk.ReadImage(str(file_path))
        return sitk.GetArrayFromImage(image).astype(np.float32)

    raise ValueError(f"Unsupported array format: {file_path}")


def load_luna16_annotations(
    path: str | Path,
    min_diameter_mm: float = 0.0,
) -> dict[str, list[dict[str, Any]]]:
    """读取 LUNA16 annotations.csv，并按 seriesuid 分组。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            diameter_mm = float(row["diameter_mm"])
            if diameter_mm < min_diameter_mm:
                continue
            record = {
                "seriesuid": row["seriesuid"],
                "coordX": float(row["coordX"]),
                "coordY": float(row["coordY"]),
                "coordZ": float(row["coordZ"]),
                "diameter_mm": diameter_mm,
            }
            grouped.setdefault(record["seriesuid"], []).append(record)
    return grouped


def extract_seriesuid_from_path(path: str | Path) -> str:
    """从 anomaly 输出文件名中恢复 seriesuid。"""
    name = Path(path).name
    for suffix in ("_anomaly.nii.gz", "_anomaly.nii", "_anomaly.npy", "_anomaly.npz", "_anomaly.mhd"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def _world_xyz_to_index_zyx(
    world_xyz: Sequence[float],
    origin_xyz: Sequence[float],
    spacing_xyz: Sequence[float],
) -> np.ndarray:
    voxel_xyz = world_to_voxel(
        np.asarray(world_xyz, dtype=np.float32),
        np.asarray(origin_xyz, dtype=np.float32),
        np.asarray(spacing_xyz, dtype=np.float32),
    )
    return np.asarray([voxel_xyz[2], voxel_xyz[1], voxel_xyz[0]], dtype=np.float32)


def _sphere_mask(
    shape: Sequence[int],
    center_zyx: Sequence[float],
    radius_mm: float,
    spacing_zyx: Sequence[float],
) -> np.ndarray:
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dz = (z - float(center_zyx[0])) * float(spacing_zyx[0])
    dy = (y - float(center_zyx[1])) * float(spacing_zyx[1])
    dx = (x - float(center_zyx[2])) * float(spacing_zyx[2])
    dist_sq = dz * dz + dy * dy + dx * dx
    return dist_sq <= float(radius_mm) ** 2


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _safe_max(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.max(np.asarray(values, dtype=np.float32)))


def _false_positive_components(binary_map: np.ndarray, nodule_mask: np.ndarray) -> int:
    outside_binary = np.logical_and(binary_map, np.logical_not(nodule_mask))
    labeled = label(outside_binary.astype(np.uint8), connectivity=1)
    return int(labeled.max())


def _build_case_context(
    anomaly_map: np.ndarray,
    reference_image: sitk.Image,
    annotations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    scores = np.asarray(anomaly_map, dtype=np.float32)
    if scores.ndim != 3:
        raise ValueError(f"Expected 3D anomaly map, got {scores.ndim}D")

    spacing_xyz = reference_image.GetSpacing()
    origin_xyz = reference_image.GetOrigin()
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    lesion_masks: list[np.ndarray] = []
    lesion_max_scores: list[float] = []
    lesion_mean_scores: list[float] = []
    nodule_mask = np.zeros(scores.shape, dtype=bool)

    for ann in annotations:
        center_zyx = _world_xyz_to_index_zyx(
            (ann["coordX"], ann["coordY"], ann["coordZ"]),
            origin_xyz,
            spacing_xyz,
        )
        lesion_mask = _sphere_mask(
            shape=scores.shape,
            center_zyx=center_zyx,
            radius_mm=float(ann["diameter_mm"]) / 2.0,
            spacing_zyx=spacing_zyx,
        )
        lesion_masks.append(lesion_mask)
        nodule_mask |= lesion_mask
        lesion_scores = scores[lesion_mask]
        if lesion_scores.size == 0:
            lesion_max_scores.append(0.0)
            lesion_mean_scores.append(0.0)
            continue
        lesion_max_scores.append(float(np.max(lesion_scores)))
        lesion_mean_scores.append(float(np.mean(lesion_scores)))

    outside_scores = scores[np.logical_not(nodule_mask)]
    peak_index = np.unravel_index(int(np.argmax(scores)), scores.shape)

    return {
        "scores": scores,
        "annotations": list(annotations),
        "nodule_mask": nodule_mask,
        "lesion_masks": lesion_masks,
        "lesion_max_scores": lesion_max_scores,
        "lesion_mean_scores": lesion_mean_scores,
        "outside_scores": outside_scores,
        "peak_index": peak_index,
    }


def _evaluate_case_context(context: dict[str, Any], threshold: float) -> dict[str, Any]:
    scores = context["scores"]
    annotations = context["annotations"]
    nodule_mask = context["nodule_mask"]
    lesion_max_scores = context["lesion_max_scores"]
    lesion_mean_scores = context["lesion_mean_scores"]
    outside_scores = context["outside_scores"]

    nodule_hits = sum(1 for value in lesion_max_scores if value >= threshold)
    binary_map = scores >= float(threshold)
    peak_hits_nodule = bool(nodule_mask[context["peak_index"]]) if annotations else False
    fp_components = _false_positive_components(binary_map, nodule_mask)

    return {
        "has_nodule": int(bool(annotations)),
        "nodule_count": int(len(annotations)),
        "nodule_hits": int(nodule_hits),
        "nodule_recall": float(nodule_hits / len(annotations)) if annotations else None,
        "peak_hits_nodule": int(peak_hits_nodule),
        "case_max_score": float(np.max(scores)),
        "case_mean_score": float(np.mean(scores)),
        "case_positive": int(bool(np.any(binary_map))),
        "nodule_max_score_mean": _safe_mean(lesion_max_scores),
        "nodule_max_score_max": _safe_max(lesion_max_scores),
        "nodule_mean_score_mean": _safe_mean(lesion_mean_scores),
        "outside_max_score": float(np.max(outside_scores)) if outside_scores.size else 0.0,
        "outside_mean_score": float(np.mean(outside_scores)) if outside_scores.size else 0.0,
        "predicted_positive_voxels": int(np.count_nonzero(binary_map)),
        "fp_components": int(fp_components),
        "threshold": float(threshold),
    }


def _summarize_case_reports(
    cases: Sequence[dict[str, Any]],
    *,
    threshold: float,
    threshold_percentile: float | None = None,
) -> dict[str, Any]:
    case_labels = [int(case["has_nodule"]) for case in cases]
    case_scores = [float(case["case_max_score"]) for case in cases]
    lesion_hits = sum(int(case["nodule_hits"]) for case in cases)
    lesion_total = sum(int(case["nodule_count"]) for case in cases)
    fp_components_total = sum(int(case["fp_components"]) for case in cases)
    positive_cases = int(sum(case_labels))
    negative_cases = int(len(cases) - positive_cases)
    peak_hits_positive = sum(int(case["peak_hits_nodule"]) for case in cases if int(case["has_nodule"]) == 1)
    fp_negative_total = sum(int(case["fp_components"]) for case in cases if int(case["has_nodule"]) == 0)

    summary: dict[str, Any] = {
        "case_count": int(len(cases)),
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
        "threshold": float(threshold),
        "lesion_total": int(lesion_total),
        "lesion_hits": int(lesion_hits),
        "lesion_recall": float(lesion_hits / lesion_total) if lesion_total > 0 else None,
        "peak_localization_rate": float(peak_hits_positive / positive_cases) if positive_cases > 0 else None,
        "fp_components_total": int(fp_components_total),
        "fp_per_case": float(fp_components_total / len(cases)),
        "fp_per_negative_case": float(fp_negative_total / negative_cases) if negative_cases > 0 else None,
        "positive_case_score_mean": float(np.mean([score for score, label in zip(case_scores, case_labels) if label == 1])) if positive_cases > 0 else None,
        "negative_case_score_mean": float(np.mean([score for score, label in zip(case_scores, case_labels) if label == 0])) if negative_cases > 0 else None,
    }
    if threshold_percentile is not None:
        summary["threshold_percentile"] = float(threshold_percentile)

    if len(set(case_labels)) >= 2:
        scores = np.asarray(case_scores, dtype=np.float32)
        labels = np.asarray(case_labels, dtype=np.uint8)
        summary["case_auc"] = float(roc_auc_score(labels, scores))
        summary["case_ap"] = float(average_precision_score(labels, scores))
    else:
        summary["case_auc"] = None
        summary["case_ap"] = None
    return summary


def evaluate_luna16_case(
    anomaly_map: np.ndarray,
    reference_image: sitk.Image,
    annotations: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """在单病例上计算 LUNA16 弱标注指标。"""
    context = _build_case_context(anomaly_map, reference_image, annotations)
    return _evaluate_case_context(context, threshold)


def evaluate_luna16_threshold_sweep(
    case_contexts: Sequence[dict[str, Any]],
    score_percentiles: Sequence[float],
) -> list[dict[str, Any]]:
    """按多个分位数阈值扫描病例/结节级指标，形成 FROC 风格摘要。"""
    if not score_percentiles:
        return []

    flat_scores = np.concatenate([np.asarray(context["scores"], dtype=np.float32).reshape(-1) for context in case_contexts])
    sweep: list[dict[str, Any]] = []
    for percentile in score_percentiles:
        threshold = float(np.percentile(flat_scores, float(percentile)))
        cases = []
        for context in case_contexts:
            report = _evaluate_case_context(context, threshold)
            report["seriesuid"] = context["seriesuid"]
            cases.append(report)
        summary = _summarize_case_reports(
            cases,
            threshold=threshold,
            threshold_percentile=float(percentile),
        )
        sweep.append(summary)
    return sweep


def evaluate_luna16_detection_dir(
    anomaly_paths: Sequence[str | Path],
    annotations_path: str | Path,
    ct_dir: str | Path,
    score_percentile: float = 99.5,
    min_diameter_mm: float = 0.0,
    score_percentiles: Sequence[float] | None = None,
) -> dict[str, Any]:
    """对 LUNA16 检测输出目录做病例级与结节级弱标注评估。"""
    if not anomaly_paths:
        raise ValueError("anomaly_paths cannot be empty")

    path_list = [Path(path) for path in anomaly_paths]
    arrays = [np.asarray(_load_array(path), dtype=np.float32) for path in path_list]
    flat_scores = np.concatenate([array.reshape(-1) for array in arrays if array.size > 0])
    if flat_scores.size == 0:
        raise ValueError("anomaly maps cannot be empty")

    threshold = float(np.percentile(flat_scores, score_percentile))
    grouped_annotations = load_luna16_annotations(annotations_path, min_diameter_mm=min_diameter_mm)
    ct_root = Path(ct_dir)

    case_contexts: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    for path, anomaly_map in zip(path_list, arrays):
        seriesuid = extract_seriesuid_from_path(path)
        reference_path = ct_root / f"{seriesuid}.mhd"
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing reference CT for {seriesuid}: {reference_path}")

        annotations = grouped_annotations.get(seriesuid, [])
        context = _build_case_context(
            anomaly_map=anomaly_map,
            reference_image=sitk.ReadImage(str(reference_path)),
            annotations=annotations,
        )
        context["seriesuid"] = seriesuid
        case_contexts.append(context)
        report = _evaluate_case_context(context, threshold)
        report["seriesuid"] = seriesuid
        cases.append(report)
    summary = _summarize_case_reports(
        cases,
        threshold=threshold,
        threshold_percentile=float(score_percentile),
    )
    sweep = evaluate_luna16_threshold_sweep(
        case_contexts,
        score_percentiles=score_percentiles or (),
    )

    return {
        "annotations_path": str(annotations_path),
        "ct_dir": str(ct_dir),
        "summary": summary,
        "cases": cases,
        "sweep": sweep,
    }
