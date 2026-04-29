"""
 * [INPUT]: 依赖 csv, json, pathlib, numpy, pytest, SimpleITK, scripts.run_experiments, src.evaluation.luna16
 * [OUTPUT]: 对外提供 LUNA16 弱标注评估与 CLI 的单元测试，覆盖阈值扫描、FROC 曲线和连通域后处理
 * [POS]: tests/unit 的论文指标回归入口，负责守住 seriesuid 解析、病例汇总和工作点导出
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
 */
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.evaluation.luna16 import (
    build_luna16_froc_curve,
    evaluate_luna16_case,
    evaluate_luna16_detection_dir,
    extract_seriesuid_from_path,
    select_luna16_operating_point,
)


def _write_reference_ct(
    path: Path,
    shape=(5, 5, 5),
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
) -> None:
    image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    sitk.WriteImage(image, str(path))


def test_extract_seriesuid_from_anomaly_path():
    assert extract_seriesuid_from_path("1.2.3_anomaly.nii.gz") == "1.2.3"


def test_evaluate_luna16_case_reports_hit_and_false_positive():
    anomaly_map = np.zeros((5, 5, 5), dtype=np.float32)
    anomaly_map[2, 2, 2] = 1.0
    anomaly_map[0, 0, 0] = 0.7

    reference = sitk.GetImageFromArray(np.zeros((5, 5, 5), dtype=np.float32))
    reference.SetSpacing((1.0, 1.0, 1.0))
    reference.SetOrigin((0.0, 0.0, 0.0))

    report = evaluate_luna16_case(
        anomaly_map=anomaly_map,
        reference_image=reference,
        annotations=[{"coordX": 2.0, "coordY": 2.0, "coordZ": 2.0, "diameter_mm": 2.0}],
        threshold=0.5,
    )

    assert report["has_nodule"] == 1
    assert report["nodule_hits"] == 1
    assert report["nodule_recall"] == pytest.approx(1.0, rel=1e-6)
    assert report["peak_hits_nodule"] == 1
    assert report["fp_components"] == 1
    assert report["case_positive"] == 1


def test_evaluate_luna16_case_postprocess_removes_small_false_positive():
    anomaly_map = np.zeros((5, 5, 5), dtype=np.float32)
    anomaly_map[2, 2, 2] = 1.0
    anomaly_map[0, 0, 0] = 0.7

    reference = sitk.GetImageFromArray(np.zeros((5, 5, 5), dtype=np.float32))
    reference.SetSpacing((1.0, 1.0, 1.0))
    reference.SetOrigin((0.0, 0.0, 0.0))

    report = evaluate_luna16_case(
        anomaly_map=anomaly_map,
        reference_image=reference,
        annotations=[{"coordX": 2.0, "coordY": 2.0, "coordZ": 2.0, "diameter_mm": 2.0}],
        threshold=0.5,
        component_min_size_voxels=2,
    )

    assert report["nodule_hits"] == 0
    assert report["fp_components"] == 0
    assert report["predicted_positive_voxels"] == 0


def test_evaluate_luna16_detection_dir_summarizes_case_and_lesion_metrics(tmp_path):
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    annotations_path = tmp_path / "annotations.csv"

    positive_series = "1.2.3"
    negative_series = "9.8.7"
    _write_reference_ct(ct_dir / f"{positive_series}.mhd")
    _write_reference_ct(ct_dir / f"{negative_series}.mhd")

    with open(annotations_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seriesuid": positive_series,
                "coordX": 2.0,
                "coordY": 2.0,
                "coordZ": 2.0,
                "diameter_mm": 2.0,
            }
        )

    positive_map = np.zeros((5, 5, 5), dtype=np.float32)
    positive_map[2, 2, 2] = 1.0
    negative_map = np.zeros((5, 5, 5), dtype=np.float32)
    negative_map[0, 0, 0] = 0.2

    positive_path = tmp_path / f"{positive_series}_anomaly.npy"
    negative_path = tmp_path / f"{negative_series}_anomaly.npy"
    np.save(positive_path, positive_map)
    np.save(negative_path, negative_map)

    report = evaluate_luna16_detection_dir(
        anomaly_paths=[positive_path, negative_path],
        annotations_path=annotations_path,
        ct_dir=ct_dir,
        score_percentile=99.0,
    )

    assert report["summary"]["case_count"] == 2
    assert report["summary"]["positive_cases"] == 1
    assert report["summary"]["negative_cases"] == 1
    assert report["summary"]["lesion_total"] == 1
    assert report["summary"]["lesion_hits"] == 1
    assert report["summary"]["lesion_recall"] == pytest.approx(1.0, rel=1e-6)
    assert report["summary"]["case_auc"] == pytest.approx(1.0, rel=1e-6)
    assert report["summary"]["case_ap"] == pytest.approx(1.0, rel=1e-6)
    assert len(report["sweep"]) == 0
    assert len(report["froc_curve"]) == 0
    assert report["recommended"] is None
    assert {case["seriesuid"] for case in report["cases"]} == {positive_series, negative_series}


def test_evaluate_luna16_threshold_sweep_returns_operating_points_and_froc_curve(tmp_path):
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    annotations_path = tmp_path / "annotations.csv"

    positive_series = "1.2.3"
    negative_series = "9.8.7"
    _write_reference_ct(ct_dir / f"{positive_series}.mhd")
    _write_reference_ct(ct_dir / f"{negative_series}.mhd")

    with open(annotations_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seriesuid": positive_series,
                "coordX": 2.0,
                "coordY": 2.0,
                "coordZ": 2.0,
                "diameter_mm": 2.0,
            }
        )

    positive_map = np.zeros((5, 5, 5), dtype=np.float32)
    positive_map[2, 2, 2] = 1.0
    negative_map = np.zeros((5, 5, 5), dtype=np.float32)
    negative_map[0, 0, 0] = 0.1

    positive_path = tmp_path / f"{positive_series}_anomaly.npy"
    negative_path = tmp_path / f"{negative_series}_anomaly.npy"
    np.save(positive_path, positive_map)
    np.save(negative_path, negative_map)

    report = evaluate_luna16_detection_dir(
        anomaly_paths=[positive_path, negative_path],
        annotations_path=annotations_path,
        ct_dir=ct_dir,
        score_percentile=99.0,
        score_percentiles=[90.0, 99.0],
    )

    assert len(report["sweep"]) == 2
    assert report["sweep"][0]["threshold_percentile"] == pytest.approx(90.0, rel=1e-6)
    assert "lesion_recall" in report["sweep"][0]
    assert "fp_per_case" in report["sweep"][0]
    assert len(report["froc_curve"]) == 2
    assert report["froc_curve"][0]["fp_per_case"] <= report["froc_curve"][1]["fp_per_case"]


def test_build_luna16_froc_curve_sorts_by_fp():
    sweep = [
        {"threshold_percentile": 99.0, "threshold": 0.9, "lesion_recall": 0.4, "fp_per_case": 10.0, "fp_per_negative_case": 10.0, "peak_localization_rate": 0.0},
        {"threshold_percentile": 99.9, "threshold": 1.2, "lesion_recall": 0.2, "fp_per_case": 2.0, "fp_per_negative_case": 2.0, "peak_localization_rate": 0.0},
    ]

    curve = build_luna16_froc_curve(sweep)

    assert [point["threshold_percentile"] for point in curve] == [99.9, 99.0]
    assert curve[0]["sensitivity"] == pytest.approx(0.2, rel=1e-6)


def test_select_luna16_operating_point_prefers_recall_then_fp():
    sweep = [
        {"threshold_percentile": 99.0, "lesion_recall": 0.5, "fp_per_case": 10.0, "case_auc": 0.9},
        {"threshold_percentile": 99.5, "lesion_recall": 0.5, "fp_per_case": 8.0, "case_auc": 0.9},
        {"threshold_percentile": 99.9, "lesion_recall": 0.25, "fp_per_case": 1.0, "case_auc": 0.95},
    ]

    chosen = select_luna16_operating_point(sweep)

    assert chosen["threshold_percentile"] == pytest.approx(99.5, rel=1e-6)


def test_experiments_cli_luna16_writes_summary(tmp_path):
    from scripts import run_experiments

    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    series = "1.2.3"
    _write_reference_ct(ct_dir / f"{series}.mhd")

    annotations_path = tmp_path / "annotations.csv"
    with open(annotations_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seriesuid": series,
                "coordX": 2.0,
                "coordY": 2.0,
                "coordZ": 2.0,
                "diameter_mm": 2.0,
            }
        )

    anomaly_dir = tmp_path / "anomaly"
    anomaly_dir.mkdir()
    anomaly_map = np.zeros((5, 5, 5), dtype=np.float32)
    anomaly_map[2, 2, 2] = 1.0
    np.save(anomaly_dir / f"{series}_anomaly.npy", anomaly_map)

    output_path = tmp_path / "luna16.json"
    report = run_experiments.main(
        [
            "--config",
            str(Path("configs/experiments.yaml")),
            "luna16",
            "--input-dir",
            str(anomaly_dir),
            "--annotations",
            str(annotations_path),
            "--ct-dir",
            str(ct_dir),
            "--output",
            str(output_path),
            "--score-percentile",
            "90",
            "--score-percentiles",
            "90",
            "95",
            "--component-min-size-voxels",
            "1",
        ]
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.exists()
    assert report["summary"]["case_count"] == 1
    assert written["summary"]["lesion_hits"] == 1
    assert len(written["froc_curve"]) == 2
