"""
 * [INPUT]: 依赖 json, pathlib, numpy, pytest, src.experiments
 * [OUTPUT]: 对外提供实验层健康统计、合成异常与 ablation 汇总测试
 * [POS]: tests/unit/ 的实验层验证器，覆盖竞赛基础实验骨架
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.experiments import (
    compare_metric_reports,
    evaluate_synthetic_sensitivity,
    inject_spherical_anomaly,
    iter_input_paths,
    load_array,
    summarize_anomaly_map_files,
    summarize_anomaly_maps,
    summarize_metric_records,
    summarize_metric_reports,
)


def test_summarize_anomaly_maps_statistics():
    maps = [
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([[2.0, 3.0]], dtype=np.float32),
    ]

    summary = summarize_anomaly_maps(maps, percentiles=(50, 95))

    assert summary["count"] == 2
    assert summary["voxels"] == 4
    assert summary["mean"] == pytest.approx(1.5, rel=1e-6)
    assert summary["std"] == pytest.approx(np.std([0.0, 1.0, 2.0, 3.0]), rel=1e-6)
    assert summary["p50"] == pytest.approx(1.5, rel=1e-6)
    assert summary["p95"] == pytest.approx(np.percentile([0.0, 1.0, 2.0, 3.0], 95), rel=1e-6)


def test_summarize_anomaly_map_files_reads_numpy(tmp_path):
    path = tmp_path / "map.npy"
    np.save(path, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    summary = summarize_anomaly_map_files([path], percentiles=(50, 95))

    assert summary["count"] == 1
    assert summary["mean"] == pytest.approx(2.5, rel=1e-6)
    assert summary["p95"] == pytest.approx(np.percentile([1.0, 2.0, 3.0, 4.0], 95), rel=1e-6)


def test_inject_spherical_anomaly_returns_mask_and_modified_volume():
    volume = np.zeros((7, 7, 7), dtype=np.float32)

    modified, mask = inject_spherical_anomaly(volume, center=(3, 3, 3), radius=1, intensity=2.5)

    assert mask.sum() == 7
    assert modified[3, 3, 3] == pytest.approx(2.5, rel=1e-6)
    assert modified[0, 0, 0] == pytest.approx(0.0, rel=1e-6)


def test_evaluate_synthetic_sensitivity_returns_structured_results():
    volume = np.zeros((7, 7, 7), dtype=np.float32)

    def score_fn(candidate: np.ndarray) -> np.ndarray:
        return candidate

    results = evaluate_synthetic_sensitivity(
        volume,
        score_fn=score_fn,
        radii=(1, 2),
        intensities=(1.0,),
        center=(3, 3, 3),
    )

    assert len(results) == 2
    assert results[0]["radius"] == 1
    assert results[0]["intensity"] == pytest.approx(1.0, rel=1e-6)
    assert results[0]["lesion_voxels"] > 0
    assert results[0]["contrast"] >= 0.0


def test_summarize_metric_records_groups_by_dataset_name():
    records = [
        {"dataset_name": "luna16", "dice": 0.8, "auc": 0.9},
        {"dataset_name": "luna16", "dice": 0.9, "auc": 0.95},
        {"dataset_name": "lidc", "dice": 0.6, "auc": 0.7},
    ]

    summary = summarize_metric_records(records, group_key="dataset_name")

    assert summary["luna16"]["count"] == 2
    assert summary["luna16"]["dice_mean"] == pytest.approx(0.85, rel=1e-6)
    assert summary["luna16"]["auc_max"] == pytest.approx(0.95, rel=1e-6)
    assert summary["lidc"]["dice_mean"] == pytest.approx(0.6, rel=1e-6)


def test_summarize_metric_reports_reads_json(tmp_path):
    first = tmp_path / "run_a.json"
    second = tmp_path / "run_b.json"
    first.write_text(json.dumps({"run_name": "a", "dice": 0.8, "auc": 0.9}), encoding="utf-8")
    second.write_text(json.dumps({"run_name": "a", "dice": 1.0, "auc": 0.8}), encoding="utf-8")

    summary = summarize_metric_reports([first, second], group_key="run_name")

    assert summary["a"]["count"] == 2
    assert summary["a"]["dice_mean"] == pytest.approx(0.9, rel=1e-6)


def test_summarize_metric_reports_accepts_utf8_bom_json(tmp_path):
    report = tmp_path / "run_bom.json"
    report.write_text(
        json.dumps({"run_name": "bom", "dice": 0.7, "auc": 0.8}),
        encoding="utf-8-sig",
    )

    summary = summarize_metric_reports([report], group_key="run_name")

    assert summary["bom"]["count"] == 1
    assert summary["bom"]["dice_mean"] == pytest.approx(0.7, rel=1e-6)


def test_compare_metric_reports_returns_deltas(tmp_path):
    baseline = tmp_path / "baseline.json"
    variant = tmp_path / "variant.json"
    baseline.write_text(json.dumps({"dice": 0.8, "auc": 0.9, "loss": 0.2}), encoding="utf-8")
    variant.write_text(json.dumps({"dice": 0.85, "auc": 0.95, "loss": 0.15}), encoding="utf-8")

    summary = compare_metric_reports(baseline, variant)

    assert summary["baseline_path"].endswith("baseline.json")
    assert summary["variant_path"].endswith("variant.json")
    assert summary["comparison"]["dice"]["baseline"] == pytest.approx(0.8, rel=1e-6)
    assert summary["comparison"]["dice"]["variant"] == pytest.approx(0.85, rel=1e-6)
    assert summary["comparison"]["dice"]["delta"] == pytest.approx(0.05, rel=1e-6)
    assert summary["comparison"]["auc"]["delta_pct"] == pytest.approx((0.95 - 0.9) / 0.9, rel=1e-6)


def test_iter_input_paths_and_load_array(tmp_path):
    arr_path = tmp_path / "sample.npy"
    np.save(arr_path, np.array([1.0], dtype=np.float32))

    paths = iter_input_paths(input_dir=tmp_path)
    assert arr_path in paths
    loaded = load_array(arr_path)
    assert loaded.tolist() == [1.0]


def test_experiments_cli_parse_and_dispatch(tmp_path, monkeypatch):
    from scripts import run_experiments

    input_path = tmp_path / "map.npy"
    np.save(input_path, np.array([1.0, 2.0], dtype=np.float32))
    output_path = tmp_path / "health.json"

    summary = run_experiments.main(
        [
            "--config",
            str(Path("configs/experiments.yaml")),
            "health",
            "--inputs",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    assert summary["count"] == 1
    assert output_path.exists()


def test_experiments_cli_compare_writes_summary(tmp_path):
    from scripts import run_experiments

    baseline = tmp_path / "baseline.json"
    variant = tmp_path / "variant.json"
    output_path = tmp_path / "compare.json"
    baseline.write_text(json.dumps({"dice": 0.8, "auc": 0.9}), encoding="utf-8")
    variant.write_text(json.dumps({"dice": 0.9, "auc": 0.95}), encoding="utf-8")

    summary = run_experiments.main(
        [
            "--config",
            str(Path("configs/experiments.yaml")),
            "compare",
            "--baseline",
            str(baseline),
            "--variant",
            str(variant),
            "--output",
            str(output_path),
        ]
    )

    assert output_path.exists()
    assert summary["comparison"]["dice"]["delta"] == pytest.approx(0.1, rel=1e-6)
