"""
 * [INPUT]: 依赖 json、pathlib、src.experiments.protocol 与 scripts.run_experiments
 * [OUTPUT]: 对外提供实验协议生成、渲染与 CLI 导出的回归测试
 * [POS]: tests/unit/ 的实验协议契约层，锁住 README 对应的可执行方案
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from pathlib import Path

from src.experiments import build_experiment_protocol, render_experiment_protocol_markdown


def test_build_experiment_protocol_contains_full_runbook():
    protocol = build_experiment_protocol()

    stage_keys = [stage["key"] for stage in protocol["stages"]]

    assert stage_keys[:4] == [
        "download_data",
        "train_smoke",
        "detect_smoke",
        "health_summary",
    ]
    assert "repeatability_sweep" in stage_keys
    assert "baseline_comparison" in stage_keys
    assert "synthetic_sensitivity" in stage_keys
    assert "ablation_aggregation" in stage_keys
    assert protocol["reproducibility"]["repeats"] == 3
    assert tuple(protocol["reproducibility"]["seeds"]) == (0, 1, 2)

    repeat_stage = next(stage for stage in protocol["stages"] if stage["key"] == "repeatability_sweep")
    assert len(repeat_stage["commands"]) == 9
    assert any("training.seed=0" in command["command"] for command in repeat_stage["commands"])
    assert any("training.seed=1" in command["command"] for command in repeat_stage["commands"])
    assert any("training.seed=2" in command["command"] for command in repeat_stage["commands"])

    compare_stage = next(stage for stage in protocol["stages"] if stage["key"] == "baseline_comparison")
    assert "scripts/run_experiments.py compare" in compare_stage["commands"][0]["command"]
    ablation_stage = next(stage for stage in protocol["stages"] if stage["key"] == "ablation_aggregation")
    assert "scripts/run_experiments.py ablation" in ablation_stage["commands"][0]["command"]


def test_render_experiment_protocol_markdown_contains_steps():
    protocol = build_experiment_protocol()
    text = render_experiment_protocol_markdown(protocol)

    assert "实验协议" in text
    assert "Step 1. 下载真实 CT" in text
    assert "GPU 训练 smoke test" in text
    assert "training.seed=0" in text
    assert "training.seed=1" in text
    assert "training.seed=2" in text
    assert "基线对比实验" in text


def test_run_experiments_plan_writes_markdown(tmp_path):
    from scripts import run_experiments

    output_path = tmp_path / "experiment_plan.md"

    protocol = run_experiments.main(
        [
            "--config",
            str(Path("configs/experiments.yaml")),
            "plan",
            "--output",
            str(output_path),
        ]
    )

    assert output_path.exists()
    assert protocol["title"] == "Lung-Diffusion-Anomaly 实验协议"
    text = output_path.read_text(encoding="utf-8")
    assert "Step 1. 下载真实 CT" in text
    assert "training.seed=0" in text
