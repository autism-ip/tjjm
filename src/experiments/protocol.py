"""
 * [INPUT]: 依赖 dataclasses、json、pathlib、typing，以及实验层配置字典
 * [OUTPUT]: 对外提供 build_experiment_protocol、render_experiment_protocol_markdown、save_experiment_protocol
 * [POS]: src/experiments/ 的实验协议层，把研究方案从 README 变成可导出的结构化产物
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.reporter import save_report


@dataclass(frozen=True)
class ExperimentCommand:
    label: str
    command: str
    outputs: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class ExperimentStage:
    key: str
    title: str
    goal: str
    commands: tuple[ExperimentCommand, ...] = ()
    outputs: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentProtocol:
    title: str
    research_goal: str
    data_policy: dict[str, Any]
    baselines: tuple[dict[str, str], ...]
    metrics: dict[str, tuple[str, ...]]
    stages: tuple[ExperimentStage, ...]
    reproducibility: dict[str, Any]
    artifacts: dict[str, str]
    notes: tuple[str, ...] = ()


DEFAULT_BASELINES: tuple[dict[str, str], ...] = (
    {"name": "Random", "purpose": "随机异常图下限参考"},
    {"name": "Intensity Threshold", "purpose": "固定 HU 阈值检测"},
    {"name": "AE", "purpose": "普通自编码器重建误差"},
    {"name": "AE + Sliding Window", "purpose": "当前主方法的最小可用版本"},
    {"name": "AE + Pretrained Encoder", "purpose": "验证预训练编码器的收益"},
    {"name": "Ablation without smoothing", "purpose": "去掉后处理平滑，检查稳定性"},
)

DEFAULT_METRICS: dict[str, tuple[str, ...]] = {
    "detection": ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"),
    "localization": ("dice", "iou", "lesion_sensitivity", "volume_fpr"),
    "runtime": ("train_time_per_epoch", "inference_time_per_case", "gpu_memory"),
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "data_root": "./data/raw/LUNA16",
    "download_subset": 2,
    "train": {
        "dataset_dir": "./data/raw/LUNA16",
        "luna16_raw_dir": "./data/raw/LUNA16",
        "batch_size": 2,
        "num_workers": 2,
        "max_epochs": 1,
        "precision": "16-mixed",
        "accelerator": "gpu",
        "devices": 1,
        "checkpoint_dir": "./data/tmp-checkpoints/gpu-smoke",
    },
    "detect": {
        "test_ct_dir": "./data/raw/LUNA16",
        "output_dir": "./data/tmp-detect",
    },
    "experiments": {
        "output_dir": "./outputs/experiments",
    },
    "reproducibility": {
        "seeds": (0, 1, 2),
        "repeats": 3,
    },
}


def _merge_settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = {
        "data_root": DEFAULT_SETTINGS["data_root"],
        "download_subset": DEFAULT_SETTINGS["download_subset"],
        "train": dict(DEFAULT_SETTINGS["train"]),
        "detect": dict(DEFAULT_SETTINGS["detect"]),
        "experiments": dict(DEFAULT_SETTINGS["experiments"]),
        "reproducibility": {
            "seeds": list(DEFAULT_SETTINGS["reproducibility"]["seeds"]),
            "repeats": DEFAULT_SETTINGS["reproducibility"]["repeats"],
        },
    }

    experiments = dict((config or {}).get("experiments", {}))
    protocol = dict(experiments.get("protocol", {}))

    if "data_root" in protocol:
        merged["data_root"] = str(protocol["data_root"])
    if "download_subset" in protocol:
        merged["download_subset"] = int(protocol["download_subset"])

    train_cfg = protocol.get("train", {})
    merged["train"].update(train_cfg)
    detect_cfg = protocol.get("detect", {})
    merged["detect"].update(detect_cfg)
    experiment_cfg = protocol.get("experiments", {})
    merged["experiments"].update(experiment_cfg)

    reproducibility_cfg = protocol.get("reproducibility", {})
    if "seeds" in reproducibility_cfg:
        merged["reproducibility"]["seeds"] = [int(seed) for seed in reproducibility_cfg["seeds"]]
    if "repeats" in reproducibility_cfg:
        merged["reproducibility"]["repeats"] = int(reproducibility_cfg["repeats"])

    return merged


def _train_command(settings: Mapping[str, Any], seed: int) -> str:
    train = settings["train"]
    checkpoint_dir = Path(train["checkpoint_dir"]) / f"seed-{seed}"
    return (
        "python scripts/train_autoencoder.py "
        f"data.dataset_dir={train['dataset_dir']} "
        f"data.luna16_raw_dir={train['luna16_raw_dir']} "
        f"data.batch_size={train['batch_size']} "
        f"data.num_workers={train['num_workers']} "
        "model.encoder_pretrained=false "
        "model.use_checkpoint=true "
        f"training.max_epochs={train['max_epochs']} "
        f"training.precision={train['precision']} "
        f"training.accelerator={train['accelerator']} "
        f"training.devices={train['devices']} "
        f"training.seed={seed} "
        f"training.checkpoint_dir={checkpoint_dir.as_posix()}"
    )


def _detect_command(settings: Mapping[str, Any], seed: int) -> str:
    train = settings["train"]
    detect = settings["detect"]
    checkpoint_path = Path(train["checkpoint_dir"]) / f"seed-{seed}" / "final.ckpt"
    output_dir = Path(detect["output_dir"]) / f"seed-{seed}"
    return (
        "python scripts/detect.py "
        f"data.test_ct_dir={detect['test_ct_dir']} "
        f"data.output_dir={output_dir.as_posix()} "
        f"model.checkpoint_path={checkpoint_path.as_posix()}"
    )


def _health_command(settings: Mapping[str, Any], seed: int) -> str:
    output_dir = Path(settings["experiments"]["output_dir"])
    detect_dir = Path(settings["detect"]["output_dir"]) / f"seed-{seed}"
    return (
        "python scripts/run_experiments.py health "
        f"--input-dir {detect_dir.as_posix()} "
        f"--output {(output_dir / f'health_summary_seed-{seed}.json').as_posix()}"
    )


def _synthetic_command(settings: Mapping[str, Any]) -> str:
    output_dir = Path(settings["experiments"]["output_dir"])
    return (
        "python scripts/run_experiments.py synthetic "
        "--volume <representative_volume.mhd> "
        f"--output {(output_dir / 'synthetic_summary.json').as_posix()}"
    )


def _ablation_command(settings: Mapping[str, Any]) -> str:
    output_dir = Path(settings["experiments"]["output_dir"])
    return (
        "python scripts/run_experiments.py ablation "
        "--reports <baseline_report.json> <variant_report.json> "
        f"--output {(output_dir / 'ablation_summary.json').as_posix()}"
    )


def _compare_command(settings: Mapping[str, Any]) -> str:
    output_dir = Path(settings["experiments"]["output_dir"])
    return (
        "python scripts/run_experiments.py compare "
        "--baseline <baseline_report.json> "
        "--variant <variant_report.json> "
        f"--output {(output_dir / 'compare_summary.json').as_posix()}"
    )


def _ablation_train_command(settings: Mapping[str, Any], ablation_cfg: Mapping[str, Any], 
                           dimension: str, value: Any, seed: int) -> str:
    """生成消融实验的训练命令"""
    train = settings["train"]
    checkpoint_dir = Path(train["checkpoint_dir"]) / f"ablation_{dimension}_{value}" / f"seed-{seed}"
    
    base_cmd = (
        "python scripts/train_autoencoder.py "
        f"data.dataset_dir={train['dataset_dir']} "
        f"data.luna16_raw_dir={train['luna16_raw_dir']} "
        f"data.batch_size={train['batch_size']} "
        f"data.num_workers={train['num_workers']} "
        f"training.max_epochs={train['max_epochs']} "
        f"training.precision={train['precision']} "
        f"training.accelerator={train['accelerator']} "
        f"training.devices={train['devices']} "
        f"training.seed={seed} "
        f"training.checkpoint_dir={checkpoint_dir.as_posix()} "
    )
    
    # 根据消融维度添加特定参数
    if dimension == "encoder":
        base_cmd += f"model.encoder_name={value} "
        if value != "swin_unetr":
            base_cmd += "model.encoder_pretrained=false "
    elif dimension == "loss":
        base_cmd += f"loss.name={value} "
    elif dimension == "patch_size":
        base_cmd += f"data.patch_size=[{value},{value},{value}] "
    elif dimension == "pretrained":
        if value == "random":
            base_cmd += "model.encoder_pretrained=false "
        elif value == "swin_ssl":
            base_cmd += "model.encoder_pretrained=true "
    elif dimension == "freeze":
        base_cmd += f"model.freeze_encoder={str(value).lower()} "
    
    return base_cmd


def _comparison_train_command(settings: Mapping[str, Any], baseline_name: str, seed: int) -> str:
    """生成对比实验的训练命令"""
    train = settings["train"]
    checkpoint_dir = Path(train["checkpoint_dir"]) / f"comparison_{baseline_name}" / f"seed-{seed}"
    
    base_cmd = (
        "python scripts/train_autoencoder.py "
        f"data.dataset_dir={train['dataset_dir']} "
        f"data.luna16_raw_dir={train['luna16_raw_dir']} "
        f"data.batch_size={train['batch_size']} "
        f"data.num_workers={train['num_workers']} "
        f"training.max_epochs={train['max_epochs']} "
        f"training.precision={train['precision']} "
        f"training.accelerator={train['accelerator']} "
        f"training.devices={train['devices']} "
        f"training.seed={seed} "
        f"training.checkpoint_dir={checkpoint_dir.as_posix()} "
    )
    
    # 根据基线类型添加特定参数
    if baseline_name == "ae_no_pretrain":
        base_cmd += "model.encoder_pretrained=false "
    elif baseline_name == "ae_pretrained":
        base_cmd += "model.encoder_pretrained=true "
    elif baseline_name == "ae_sliding_window":
        base_cmd += "model.encoder_pretrained=true "
    elif baseline_name == "ae_postprocess":
        base_cmd += "model.encoder_pretrained=true "
    
    return base_cmd


def _robustness_train_command(settings: Mapping[str, Any], dimension: str, value: Any, seed: int) -> str:
    """生成鲁棒性实验的训练命令"""
    train = settings["train"]
    checkpoint_dir = Path(train["checkpoint_dir"]) / f"robustness_{dimension}_{value}" / f"seed-{seed}"
    
    base_cmd = (
        "python scripts/train_autoencoder.py "
        f"data.dataset_dir={train['dataset_dir']} "
        f"data.luna16_raw_dir={train['luna16_raw_dir']} "
        f"data.batch_size={train['batch_size']} "
        f"data.num_workers={train['num_workers']} "
        f"training.max_epochs={train['max_epochs']} "
        f"training.precision={train['precision']} "
        f"training.accelerator={train['accelerator']} "
        f"training.devices={train['devices']} "
        f"training.seed={seed} "
        f"training.checkpoint_dir={checkpoint_dir.as_posix()} "
    )
    
    # 根据鲁棒性维度添加特定参数
    if dimension == "data_ratio":
        base_cmd += f"data.train_ratio={value} "
    elif dimension == "noise":
        base_cmd += f"data.noise_sigma={value} "
    elif dimension == "spacing":
        base_cmd += f"data.target_spacing=[{value},{value},{value}] "
    elif dimension == "hu_window":
        base_cmd += f"data.hu_min={value[0]} data.hu_max={value[1]} "
    
    return base_cmd


def _build_stages(settings: Mapping[str, Any]) -> tuple[ExperimentStage, ...]:
    seeds = settings["reproducibility"]["seeds"]
    download_subset = settings["download_subset"]
    data_root = settings["data_root"]
    
    # 获取新增的实验配置
    comparison_cfg = settings.get("comparison", {})
    ablation_cfg = settings.get("ablation", {})
    sensitivity_cfg = settings.get("sensitivity", {})
    robustness_cfg = settings.get("robustness", {})
    efficiency_cfg = settings.get("efficiency", {})

    repeat_commands = tuple(
        ExperimentCommand(
            label=f"seed={seed}",
            command=_train_command(settings, seed),
            outputs=(str((Path(settings["train"]["checkpoint_dir"]) / f"seed-{seed}" / "final.ckpt").as_posix()),),
            notes="先训练，再用同一 seed 的 checkpoint 跑检测与统计。",
        )
        for seed in seeds
    )

    repeat_detection = tuple(
        ExperimentCommand(
            label=f"seed={seed}",
            command=_detect_command(settings, seed),
            outputs=(str((Path(settings["detect"]["output_dir"]) / f"seed-{seed}").as_posix()),),
            notes="检测输出与训练 seed 一一对应。",
        )
        for seed in seeds
    )

    repeat_health = tuple(
        ExperimentCommand(
            label=f"seed={seed}",
            command=_health_command(settings, seed),
            outputs=(
                str((Path(settings["experiments"]["output_dir"]) / f"health_summary_seed-{seed}.json").as_posix()),
            ),
            notes="把单次检测输出汇总成实验摘要。",
        )
        for seed in seeds
    )

    # ===== 对比实验命令 =====
    comparison_baselines = comparison_cfg.get("baselines", [])
    comparison_commands = []
    for baseline in comparison_baselines:
        name = baseline["name"]
        if name in ["random", "intensity_threshold"]:
            # 这些基线不需要训练，直接在检测/评估阶段处理
            continue
        comparison_commands.append(
            ExperimentCommand(
                label=f"train_{name}",
                command=_comparison_train_command(settings, name, seeds[0]),
                outputs=(str((Path(settings["train"]["checkpoint_dir"]) / f"comparison_{name}" / f"seed-{seeds[0]}" / "final.ckpt").as_posix()),),
                notes=baseline.get("description", ""),
            )
        )

    # ===== 消融实验命令 =====
    ablation_dimensions = ["encoder", "loss", "patch_size", "overlap", "pretrained", "freeze", "decoder_layers"]
    ablation_commands = []
    for dim in ablation_dimensions:
        dim_cfg = ablation_cfg.get(dim, {})
        options = dim_cfg.get("options", [])
        default = dim_cfg.get("default")
        for value in options:
            if value == default:
                continue  # 跳过默认值（已在主实验中）
            ablation_commands.append(
                ExperimentCommand(
                    label=f"ablation_{dim}={value}",
                    command=_ablation_train_command(settings, ablation_cfg, dim, value, seeds[0]),
                    outputs=(str((Path(settings["train"]["checkpoint_dir"]) / f"ablation_{dim}_{value}" / f"seed-{seeds[0]}" / "final.ckpt").as_posix()),),
                    notes=f"消融实验: {dim}={value}",
                )
            )

    # ===== 鲁棒性实验命令 =====
    robustness_commands = []
    for dim in ["data_ratio", "noise", "spacing", "hu_window"]:
        dim_cfg = robustness_cfg.get(dim, {})
        options = dim_cfg.get("options", dim_cfg.get("sigma_list", []))
        for value in options:
            robustness_commands.append(
                ExperimentCommand(
                    label=f"robustness_{dim}={value}",
                    command=_robustness_train_command(settings, dim, value, seeds[0]),
                    outputs=(str((Path(settings["train"]["checkpoint_dir"]) / f"robustness_{dim}_{value}" / f"seed-{seeds[0]}" / "final.ckpt").as_posix()),),
                    notes=f"鲁棒性实验: {dim}={value}",
                )
            )

    return (
        ExperimentStage(
            key="download_data",
            title="下载真实 CT",
            goal="只下载少量真实 LUNA16 CT 配对，先验证数据链路。",
            commands=(
                ExperimentCommand(
                    label="subset",
                    command=f"python scripts/download_data.py --subset {download_subset}",
                    outputs=(f"{Path(data_root) / 'annotations.csv'}",),
                    notes="默认只拉少量真实病例，不做整包下载。",
                ),
            ),
            outputs=(
                f"{Path(data_root) / 'annotations.csv'}",
                f"{Path(data_root) / '<case>.mhd'}",
                f"{Path(data_root) / '<case>.raw'}",
            ),
            success_criteria=("目录里可见 annotations.csv 和少量真实 CT 配对",),
        ),
        ExperimentStage(
            key="train_smoke",
            title="GPU 训练 smoke test",
            goal="用真实 CT 跑通 1 个 epoch，验证训练链路和 checkpoint 保存。",
            commands=(repeat_commands[0],),
            outputs=(repeat_commands[0].outputs[0],),
            success_criteria=("final.ckpt 成功生成", "训练日志里没有未处理异常"),
        ),
        ExperimentStage(
            key="detect_smoke",
            title="目录级检测 smoke test",
            goal="用真实 checkpoint 在真实 CT 上跑通目录推理。",
            commands=(repeat_detection[0],),
            outputs=(repeat_detection[0].outputs[0],),
            success_criteria=("异常热图和可视化图片都生成",),
        ),
        ExperimentStage(
            key="health_summary",
            title="健康统计汇总",
            goal="把检测结果压成可写入论文的统计摘要。",
            commands=(repeat_health[0],),
            outputs=(repeat_health[0].outputs[0],),
            success_criteria=("health_summary.json 可直接读取",),
        ),
        ExperimentStage(
            key="repeatability_sweep",
            title="复现性评估",
            goal="用多个随机种子重复训练、检测和统计，估计方差和稳定性。",
            commands=repeat_commands + repeat_detection + repeat_health,
            outputs=tuple(
                command.outputs[0]
                for command in (repeat_commands + repeat_detection + repeat_health)
            ),
            success_criteria=(
                "每个 seed 都能生成独立 checkpoint、检测输出和摘要",
                "均值与标准差可以写入结果表",
                "方差 < 0.05",
            ),
            notes=("至少重复 3 次，使用不同 seed。",),
        ),
        ExperimentStage(
            key="baseline_comparison",
            title="对比实验",
            goal="把主方法和多种基线方法做同表比较。",
            commands=tuple(comparison_commands) + (
                ExperimentCommand(
                    label="compare",
                    command=_compare_command(settings),
                    outputs=((Path(settings["experiments"]["output_dir"]) / "compare_summary.json").as_posix(),),
                    notes="比较时至少保留一个 baseline 和一个 variant 报告。",
                ),
            ),
            outputs=((Path(settings["experiments"]["output_dir"]) / "compare_summary.json").as_posix(),),
            success_criteria=("compare_summary.json 中包含所有基线对比结果",),
            notes=(
                "对比实验覆盖: Random、Intensity Threshold、AE(无预训练)、AE+预训练、AE+滑窗、AE+后处理。",
                "如果方法变化很多，先做 pairwise compare，再做总表汇总。",
            ),
        ),
        ExperimentStage(
            key="ablation_study",
            title="消融实验",
            goal="量化各组件的贡献，回答'哪一块贡献最大'。",
            commands=tuple(ablation_commands) + (
                ExperimentCommand(
                    label="ablation_summary",
                    command=_ablation_command(settings),
                    outputs=((Path(settings["experiments"]["output_dir"]) / "ablation_summary.json").as_posix(),),
                    notes="输入应来自已经跑完的 baseline 与变体报告。",
                ),
            ),
            outputs=((Path(settings["experiments"]["output_dir"]) / "ablation_summary.json").as_posix(),),
            success_criteria=("ablation_summary.json 可直接用于绘表",),
            notes=(
                "消融维度: 编码器架构、损失函数、Patch尺寸、重叠率、预训练策略、冻结策略、解码器深度。",
                "每个维度只改变一个变量，其他保持默认。",
            ),
        ),
        ExperimentStage(
            key="synthetic_sensitivity",
            title="合成敏感性实验",
            goal="验证对人工注入异常的响应曲线是否单调、是否稳定。",
            commands=(
                ExperimentCommand(
                    label="synthetic",
                    command=_synthetic_command(settings),
                    outputs=((Path(settings["experiments"]["output_dir"]) / "synthetic_summary.json").as_posix(),),
                    notes="把单个代表性体积的异常响应曲线导出。",
                ),
            ),
            outputs=((Path(settings["experiments"]["output_dir"]) / "synthetic_summary.json").as_posix(),),
            success_criteria=("synthetic_summary.json 中包含不同半径和强度的曲线结果",),
            notes=(
                "敏感性参数: 半径[4,6,8,12,16]、强度[0.3,0.5,1.0,2.0]、形状[sphere,cube]。",
                "验证检测灵敏度随异常大小/强度的变化趋势。",
            ),
        ),
        ExperimentStage(
            key="threshold_sensitivity",
            title="阈值敏感性实验",
            goal="评估不同阈值方法和参数对检测性能的影响。",
            commands=(
                ExperimentCommand(
                    label="threshold_sweep",
                    command=(
                        "python scripts/run_experiments.py luna16 "
                        f"--anomaly-dir {settings['detect']['output_dir']}/seed-{seeds[0]} "
                        f"--annotations {settings.get('luna16', {}).get('annotations', './data/raw/LUNA16/annotations.csv')} "
                        f"--output {(Path(settings['experiments']['output_dir']) / 'threshold_sweep.json').as_posix()}"
                    ),
                    outputs=((Path(settings["experiments"]["output_dir"]) / "threshold_sweep.json").as_posix(),),
                    notes="在多个百分位阈值下评估结节召回率和假阳性数。",
                ),
            ),
            outputs=((Path(settings["experiments"]["output_dir"]) / "threshold_sweep.json").as_posix(),),
            success_criteria=("包含多个阈值下的召回率和假阳性数",),
            notes=(
                "阈值百分位: [99.0, 99.5, 99.9]",
                "生成FROC曲线，选择推荐工作点。",
            ),
        ),
        ExperimentStage(
            key="postprocess_sensitivity",
            title="后处理敏感性实验",
            goal="评估连通域后处理参数对检测性能的影响。",
            commands=(
                ExperimentCommand(
                    label="postprocess_sweep",
                    command=(
                        "python scripts/run_experiments.py luna16 "
                        f"--anomaly-dir {settings['detect']['output_dir']}/seed-{seeds[0]} "
                        f"--annotations {settings.get('luna16', {}).get('annotations', './data/raw/LUNA16/annotations.csv')} "
                        f"--min-voxels 50 --keep-largest "
                        f"--output {(Path(settings['experiments']['output_dir']) / 'postprocess_sweep.json').as_posix()}"
                    ),
                    outputs=((Path(settings["experiments"]["output_dir"]) / "postprocess_sweep.json").as_posix(),),
                    notes="比较不同最小连通域大小和是否保留最大连通域的效果。",
                ),
            ),
            outputs=((Path(settings["experiments"]["output_dir"]) / "postprocess_sweep.json").as_posix(),),
            success_criteria=("包含不同后处理参数下的性能对比",),
            notes=(
                "后处理参数: min_voxels=[0,10,50,100], keep_largest=[true,false]",
                "验证后处理对假阳性的抑制效果。",
            ),
        ),
        ExperimentStage(
            key="robustness_study",
            title="鲁棒性实验",
            goal="评估方法在不同数据条件下的稳定性。",
            commands=tuple(robustness_commands),
            outputs=tuple(
                str((Path(settings["train"]["checkpoint_dir"]) / f"robustness_{dim}_{value}" / f"seed-{seeds[0]}" / "final.ckpt").as_posix())
                for dim in ["data_ratio", "noise", "spacing", "hu_window"]
                for value in robustness_cfg.get(dim, {}).get("options", robustness_cfg.get(dim, {}).get("sigma_list", []))
            ),
            success_criteria=(
                "各条件下方法性能变化在可接受范围内",
                "识别方法的敏感因素和鲁棒因素",
            ),
            notes=(
                "鲁棒性维度: 数据量[25%,50%,75%,100%]、噪声[σ=0.01,0.05,0.1]、分辨率[0.5,1.0,2.0mm]、HU窗[-1000,400],[-1000,200],[-500,400]。",
                "每个维度只改变一个变量，其他保持默认。",
            ),
        ),
        ExperimentStage(
            key="efficiency_analysis",
            title="效率分析",
            goal="评估方法的计算资源需求和效率。",
            commands=(
                ExperimentCommand(
                    label="efficiency_benchmark",
                    command=(
                        "python scripts/run_experiments.py efficiency "
                        f"--checkpoint {settings['train']['checkpoint_dir']}/seed-{seeds[0]}/final.ckpt "
                        f"--output {(Path(settings['experiments']['output_dir']) / 'efficiency_report.json').as_posix()}"
                    ),
                    outputs=((Path(settings["experiments"]["output_dir"]) / "efficiency_report.json").as_posix(),),
                    notes="测量训练时间、推理时间、GPU显存、模型参数量。",
                ),
            ),
            outputs=((Path(settings["experiments"]["output_dir"]) / "efficiency_report.json").as_posix(),),
            success_criteria=("包含完整的效率指标",),
            notes=(
                "效率指标: 训练时间/epoch、推理时间/case、GPU显存峰值、模型参数量、Checkpoint大小。",
                "比较不同batch_size和patch_size下的效率。",
            ),
        ),
        ExperimentStage(
            key="final_report",
            title="最终报告生成",
            goal="汇总所有实验结果，生成可提交的实验报告。",
            commands=(
                ExperimentCommand(
                    label="generate_report",
                    command=(
                        "python scripts/run_experiments.py plan "
                        f"--output {(Path(settings['experiments']['output_dir']) / 'final_report.md').as_posix()}"
                    ),
                    outputs=((Path(settings["experiments"]["output_dir"]) / "final_report.md").as_posix(),),
                    notes="生成包含所有实验结果的Markdown报告。",
                ),
            ),
            outputs=((Path(settings["experiments"]["output_dir"]) / "final_report.md").as_posix(),),
            success_criteria=("报告包含所有实验阶段的结果和结论",),
            notes=(
                "报告应包含: 研究目标、方法描述、实验设置、结果对比、消融分析、敏感性分析、鲁棒性分析、效率分析、结论。",
                "所有图表和表格应可直接用于论文。",
            ),
        ),
    )
        for seed in seeds
    )

    repeat_detection = tuple(
        ExperimentCommand(
            label=f"seed={seed}",
            command=_detect_command(settings, seed),
            outputs=(str((Path(settings["detect"]["output_dir"]) / f"seed-{seed}").as_posix()),),
            notes="检测输出与训练 seed 一一对应。",
        )
        for seed in seeds
    )

    repeat_health = tuple(
        ExperimentCommand(
            label=f"seed={seed}",
            command=_health_command(settings, seed),
            outputs=(
                str((Path(settings["experiments"]["output_dir"]) / f"health_summary_seed-{seed}.json").as_posix()),
            ),
            notes="把单次检测输出汇总成实验摘要。",
        )
        for seed in seeds
    )

    return (
        ExperimentStage(
            key="download_data",
            title="下载真实 CT",
            goal="只下载少量真实 LUNA16 CT 配对，先验证数据链路。",
            commands=(
                ExperimentCommand(
                    label="subset",
                    command=f"python scripts/download_data.py --subset {download_subset}",
                    outputs=(f"{Path(data_root) / 'annotations.csv'}",),
                    notes="默认只拉少量真实病例，不做整包下载。",
                ),
            ),
            outputs=(
                f"{Path(data_root) / 'annotations.csv'}",
                f"{Path(data_root) / '<case>.mhd'}",
                f"{Path(data_root) / '<case>.raw'}",
            ),
            success_criteria=("目录里可见 annotations.csv 和少量真实 CT 配对",),
        ),
        ExperimentStage(
            key="train_smoke",
            title="GPU 训练 smoke test",
            goal="用真实 CT 跑通 1 个 epoch，验证训练链路和 checkpoint 保存。",
            commands=(repeat_commands[0],),
            outputs=(repeat_commands[0].outputs[0],),
            success_criteria=("final.ckpt 成功生成", "训练日志里没有未处理异常"),
        ),
        ExperimentStage(
            key="detect_smoke",
            title="目录级检测 smoke test",
            goal="用真实 checkpoint 在真实 CT 上跑通目录推理。",
            commands=(repeat_detection[0],),
            outputs=(repeat_detection[0].outputs[0],),
            success_criteria=("异常热图和可视化图片都生成",),
        ),
        ExperimentStage(
            key="health_summary",
            title="健康统计汇总",
            goal="把检测结果压成可写入论文的统计摘要。",
            commands=(repeat_health[0],),
            outputs=(repeat_health[0].outputs[0],),
            success_criteria=("health_summary.json 可直接读取",),
        ),
        ExperimentStage(
            key="repeatability_sweep",
            title="重复性评估",
            goal="用多个随机种子重复训练、检测和统计，估计方差和稳定性。",
            commands=repeat_commands + repeat_detection + repeat_health,
            outputs=tuple(
                command.outputs[0]
                for command in (repeat_commands + repeat_detection + repeat_health)
            ),
            success_criteria=(
                "每个 seed 都能生成独立 checkpoint、检测输出和摘要",
                "均值与标准差可以写入结果表",
            ),
            notes=("建议至少重复 3 次，使用不同 seed。",),
        ),
        ExperimentStage(
            key="baseline_comparison",
            title="基线对比实验",
            goal="把主方法和随机、阈值、自编码器等基线做同表比较。",
            commands=(
                ExperimentCommand(
                    label="compare",
                    command=_compare_command(settings),
                    outputs=((Path(settings["experiments"]["output_dir"]) / "compare_summary.json").as_posix(),),
                    notes="比较时至少保留一个 baseline 和一个 variant 报告。",
                ),
            ),
            outputs=((Path(settings["experiments"]["output_dir"]) / "compare_summary.json").as_posix(),),
            success_criteria=("compare_summary.json 中包含 baseline、variant、delta 与 delta_pct",),
            notes=(
                "对比实验应该覆盖 Random、Intensity Threshold、AE、AE + Sliding Window、AE + Pretrained Encoder。",
                "如果方法变化很多，先做 pairwise compare，再做总表汇总。",
            ),
        ),
        ExperimentStage(
            key="synthetic_sensitivity",
            title="合成敏感性实验",
            goal="验证对人工注入异常的响应曲线是否单调、是否稳定。",
            commands=(
                ExperimentCommand(
                    label="synthetic",
                    command=_synthetic_command(settings),
                    outputs=((Path(settings["experiments"]["output_dir"]) / "synthetic_summary.json").as_posix(),),
                    notes="把单个代表性体积的异常响应曲线导出。",
                ),
            ),
            outputs=((Path(settings["experiments"]["output_dir"]) / "synthetic_summary.json").as_posix(),),
            success_criteria=("synthetic_summary.json 中包含不同半径和强度的曲线结果",),
        ),
        ExperimentStage(
            key="ablation_aggregation",
            title="消融汇总",
            goal="把不同方法或不同设置的 metrics 报告聚合成消融表。",
            commands=(
                ExperimentCommand(
                    label="ablation",
                    command=_ablation_command(settings),
                    outputs=((Path(settings["experiments"]["output_dir"]) / "ablation_summary.json").as_posix(),),
                    notes="输入应来自已经跑完的 baseline 与变体报告。",
                ),
            ),
            outputs=((Path(settings["experiments"]["output_dir"]) / "ablation_summary.json").as_posix(),),
            success_criteria=("ablation_summary.json 可直接用于绘表",),
            notes=(
                "消融实验建议围绕预训练编码器、滑窗策略、后处理和平滑强度展开。",
                "如果要写论文，消融表应该至少能回答“哪一块贡献最大”。",
            ),
        ),
    )


def build_experiment_protocol(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    settings = _merge_settings(config)
    protocol = ExperimentProtocol(
        title="Lung-Diffusion-Anomaly 实验协议",
        research_goal="仅用正常 CT 训练重建模型，并验证其对异常区域的定位与检测能力。",
        data_policy={
            "train": "只使用正常 CT 训练，学习正常结构。",
            "validation": "正常 CT + 少量异常 CT，用于调参。",
            "test": "独立病例，不参与调参。",
            "split_rules": (
                "病例级划分，避免切片泄漏。",
                "3D CT 不可在切片级随机打散后再划分。",
                "如有多中心数据，优先做外部测试。",
            ),
        },
        baselines=DEFAULT_BASELINES,
        metrics=DEFAULT_METRICS,
        stages=_build_stages(settings),
        reproducibility={
            "seeds": tuple(settings["reproducibility"]["seeds"]),
            "repeats": settings["reproducibility"]["repeats"],
            "report_rule": (
                "每个设置至少重复 3 次。",
                "主结果报告均值和标准差。",
                "必要时补 bootstrap 置信区间。",
            ),
        },
        artifacts={
            "train_checkpoint_dir": str(Path(settings["train"]["checkpoint_dir"])),
            "detect_output_dir": str(Path(settings["detect"]["output_dir"])),
            "experiment_output_dir": str(Path(settings["experiments"]["output_dir"])),
        },
        notes=(
            "这是当前仓库可执行的科研跑法，不是纸面模板。",
            "若数据规模扩大，应先补完整的数据集划分，再重复执行这份协议。",
        ),
    )
    return asdict(protocol)


def _render_list(items: Sequence[str], indent: str = "  ") -> list[str]:
    return [f"{indent}- {item}" for item in items]


def render_experiment_protocol_markdown(protocol: Mapping[str, Any]) -> str:
    lines: list[str] = [f"# {protocol['title']}", ""]
    lines.append("## 研究目标")
    lines.append(protocol["research_goal"])
    lines.append("")

    lines.append("## 数据划分")
    for key in ("train", "validation", "test"):
        lines.append(f"- {key}：{protocol['data_policy'][key]}")
    for rule in protocol["data_policy"]["split_rules"]:
        lines.append(f"- 规则：{rule}")
    lines.append("")

    lines.append("## 对照组")
    for baseline in protocol["baselines"]:
        lines.append(f"- {baseline['name']}：{baseline['purpose']}")
    lines.append("")

    lines.append("## 评估指标")
    for group_name, metric_names in protocol["metrics"].items():
        lines.append(f"### {group_name}")
        for metric_name in metric_names:
            lines.append(f"- {metric_name}")
        lines.append("")

    lines.append("## 推荐执行顺序")
    for index, stage in enumerate(protocol["stages"], start=1):
        lines.append(f"### Step {index}. {stage['title']}")
        lines.append(f"目的：{stage['goal']}")
        lines.append("")
        lines.append("命令：")
        for command in stage["commands"]:
            lines.append(f"#### {command['label']}")
            lines.append("```powershell")
            lines.append(command["command"])
            lines.append("```")
            if command.get("notes"):
                lines.append(f"- 说明：{command['notes']}")
            if command.get("outputs"):
                lines.append("- 产物：")
                lines.extend(_render_list(command["outputs"], indent="  "))
            lines.append("")
        if stage.get("outputs"):
            lines.append("阶段输出：")
            lines.extend(_render_list(stage["outputs"], indent="  "))
        if stage.get("success_criteria"):
            lines.append("成功标准：")
            lines.extend(_render_list(stage["success_criteria"], indent="  "))
        if stage.get("notes"):
            lines.append("补充说明：")
            lines.extend(_render_list(stage["notes"], indent="  "))
        lines.append("")

    lines.append("## 重复性与统计")
    for item in protocol["reproducibility"]["report_rule"]:
        lines.append(f"- {item}")
    lines.append(f"- seeds：{', '.join(str(seed) for seed in protocol['reproducibility']['seeds'])}")
    lines.append(f"- repeats：{protocol['reproducibility']['repeats']}")
    lines.append("")

    lines.append("## 产物目录")
    for key, value in protocol["artifacts"].items():
        lines.append(f"- {key}：{value}")
    lines.append("")

    lines.append("## 备注")
    for note in protocol["notes"]:
        lines.append(f"- {note}")

    lines.append("")
    return "\n".join(lines)


def save_experiment_protocol(
    protocol: Mapping[str, Any],
    output_path: str | Path,
    format: str = "md",
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        save_report(dict(protocol), path)
        return

    text = render_experiment_protocol_markdown(protocol)
    path.write_text(text, encoding="utf-8")
