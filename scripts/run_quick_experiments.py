#!/usr/bin/env python3
"""
简化版实验执行脚本 - 快速验证流程
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class QuickExperimentRunner:
    """快速实验执行器"""
    
    DATA_ROOT = "/root/autodl-tmp/data/raw/LUNA16"
    OUTPUT_BASE = "/root/tjjm/outputs/experiments"
    CHECKPOINT_BASE = "/root/tjjm/outputs/checkpoints"
    
    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(self.OUTPUT_BASE).mkdir(parents=True, exist_ok=True)
        Path(self.CHECKPOINT_BASE).mkdir(parents=True, exist_ok=True)
        
    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def run_command(self, cmd: str, description: str = "") -> bool:
        if description:
            self.log(f"执行: {description}")
        self.log(f"命令: {cmd[:150]}...")
        
        try:
            result = subprocess.run(
                cmd, shell=True, check=True,
                capture_output=True, text=True,
                cwd=str(PROJECT_ROOT),
                timeout=3600  # 1小时超时
            )
            self.log("✓ 成功")
            return True
        except subprocess.TimeoutExpired:
            self.log("✗ 超时")
            return False
        except subprocess.CalledProcessError as e:
            self.log(f"✗ 失败: {e.stderr[:300] if e.stderr else 'Unknown error'}")
            return False
    
    def run_smoke_test(self) -> bool:
        """Smoke Test - 使用小数据集快速验证"""
        self.log("=" * 60)
        self.log("Smoke Test (快速验证)")
        self.log("=" * 60)
        
        # 训练配置 - 使用小数据集
        train_cmd = (
            "python scripts/train_autoencoder.py "
            f"data.dataset_dir={self.DATA_ROOT} "
            f"data.luna16_raw_dir={self.DATA_ROOT} "
            "data.batch_size=2 "
            "data.num_workers=2 "
            "model.encoder_name=swin_unetr "
            "model.encoder_pretrained=true "
            "model.freeze_encoder=true "
            "loss.name=weighted_mse "
            "training.max_epochs=1 "
            "training.precision=16-mixed "
            "training.accelerator=gpu "
            "training.devices=1 "
            "training.seed=42 "
            f"training.checkpoint_dir={self.CHECKPOINT_BASE}/smoke_test"
        )
        
        if not self.run_command(train_cmd, "训练 Smoke Test"):
            return False
        
        # 检查checkpoint
        checkpoint = Path(self.CHECKPOINT_BASE) / "smoke_test" / "final.ckpt"
        if not checkpoint.exists():
            self.log("Checkpoint 未生成")
            return False
        
        self.log(f"✓ Smoke Test 通过，Checkpoint: {checkpoint}")
        return True
    
    def run_baseline_training(self) -> Path:
        """基线训练"""
        self.log("=" * 60)
        self.log("基线训练 (SwinUNETR + 预训练)")
        self.log("=" * 60)
        
        checkpoint_dir = Path(self.CHECKPOINT_BASE) / "baseline"
        
        train_cmd = (
            "python scripts/train_autoencoder.py "
            f"data.dataset_dir={self.DATA_ROOT} "
            f"data.luna16_raw_dir={self.DATA_ROOT} "
            "data.batch_size=4 "
            "data.num_workers=4 "
            "model.encoder_name=swin_unetr "
            "model.encoder_pretrained=true "
            "model.freeze_encoder=true "
            "loss.name=weighted_mse "
            "training.max_epochs=10 "
            "training.precision=16-mixed "
            "training.accelerator=gpu "
            "training.devices=1 "
            "training.seed=42 "
            f"training.checkpoint_dir={checkpoint_dir}"
        )
        
        if not self.run_command(train_cmd, "基线训练"):
            return None
        
        checkpoint = checkpoint_dir / "final.ckpt"
        if checkpoint.exists():
            self.log(f"✓ 基线训练完成: {checkpoint}")
            return checkpoint
        return None
    
    def run_detection(self, checkpoint_path: Path, run_name: str) -> Path:
        """执行检测"""
        self.log("=" * 60)
        self.log(f"检测: {run_name}")
        self.log("=" * 60)
        
        output_dir = Path(self.OUTPUT_BASE) / "detection" / run_name
        
        detect_cmd = (
            "python scripts/detect.py "
            f"data.test_ct_dir={self.DATA_ROOT} "
            f"data.output_dir={output_dir} "
            f"model.checkpoint_path={checkpoint_path} "
            "detection.patch_size=[64,64,64] "
            "detection.stride=[32,32,32] "
            "detection.batch_size=8"
        )
        
        if self.run_command(detect_cmd, f"检测 {run_name}"):
            return output_dir
        return None
    
    def run_evaluation(self, detection_dir: Path, run_name: str) -> dict:
        """执行评估"""
        self.log("=" * 60)
        self.log(f"评估: {run_name}")
        self.log("=" * 60)
        
        output_dir = Path(self.OUTPUT_BASE) / "evaluation" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        eval_cmd = (
            "python scripts/run_experiments.py luna16 "
            f"--anomaly-dir {detection_dir} "
            f"--annotations {self.DATA_ROOT}/annotations.csv "
            f"--output {output_dir}/eval.json"
        )
        
        if self.run_command(eval_cmd, f"评估 {run_name}"):
            eval_file = output_dir / "eval.json"
            if eval_file.exists():
                with open(eval_file, "r") as f:
                    return json.load(f)
        return {}
    
    def run_ablation(self, dimension: str, value: str, param: str) -> Path:
        """消融实验"""
        self.log("=" * 60)
        self.log(f"消融: {dimension}={value}")
        self.log("=" * 60)
        
        checkpoint_dir = Path(self.CHECKPOINT_BASE) / f"ablation_{dimension}_{value}"
        
        train_cmd = (
            "python scripts/train_autoencoder.py "
            f"data.dataset_dir={self.DATA_ROOT} "
            f"data.luna16_raw_dir={self.DATA_ROOT} "
            "data.batch_size=4 "
            "data.num_workers=4 "
            f"{param}={value} "
            "loss.name=weighted_mse "
            "training.max_epochs=5 "
            "training.precision=16-mixed "
            "training.accelerator=gpu "
            "training.devices=1 "
            "training.seed=42 "
            f"training.checkpoint_dir={checkpoint_dir}"
        )
        
        if self.run_command(train_cmd, f"消融 {dimension}={value}"):
            checkpoint = checkpoint_dir / "final.ckpt"
            if checkpoint.exists():
                return checkpoint
        return None
    
    def generate_report(self, results: dict):
        """生成报告"""
        self.log("=" * 60)
        self.log("生成最终报告")
        self.log("=" * 60)
        
        report = []
        report.append("# Lung-Diffusion-Anomaly 实验报告")
        report.append("")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 基线结果
        if "baseline" in results:
            report.append("## 基线结果")
            report.append("")
            baseline = results["baseline"]
            report.append(f"- **方法**: SwinUNETR + WeightedMSELoss")
            report.append(f"- **结节召回率**: {baseline.get('lesion_recall', 'N/A')}")
            report.append(f"- **假阳性/case**: {baseline.get('fp_per_case', 'N/A')}")
            report.append(f"- **AUC**: {baseline.get('case_auc', 'N/A')}")
            report.append("")
        
        # 消融结果
        if "ablation" in results:
            report.append("## 消融实验结果")
            report.append("")
            report.append("| 维度 | 值 | 结节召回率 | 假阳性/case | AUC |")
            report.append("|------|-----|-----------|------------|-----|")
            for item in results["ablation"]:
                report.append(f"| {item.get('dimension', 'N/A')} | {item.get('value', 'N/A')} | {item.get('lesion_recall', 'N/A')} | {item.get('fp_per_case', 'N/A')} | {item.get('case_auc', 'N/A')} |")
            report.append("")
        
        # 保存报告
        report_file = Path(self.OUTPUT_BASE) / f"report_{self.run_id}.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report))
        
        self.log(f"✓ 报告已生成: {report_file}")
    
    def run_all(self):
        """执行所有实验"""
        self.log("=" * 60)
        self.log("开始执行快速实验")
        self.log("=" * 60)
        
        start_time = time.time()
        results = {"ablation": []}
        
        # 1. Smoke Test
        if not self.run_smoke_test():
            self.log("Smoke Test 失败，终止实验")
            return
        
        # 2. 基线训练
        baseline_checkpoint = self.run_baseline_training()
        if not baseline_checkpoint:
            self.log("基线训练失败，终止实验")
            return
        
        # 3. 基线检测
        detect_dir = self.run_detection(baseline_checkpoint, "baseline")
        if detect_dir:
            eval_result = self.run_evaluation(detect_dir, "baseline")
            if eval_result:
                results["baseline"] = eval_result
        
        # 4. 消融实验
        ablation_configs = [
            ("loss", "mse", "loss.name"),
            ("encoder", "resnet", "model.encoder_name"),
        ]
        
        for dimension, value, param in ablation_configs:
            checkpoint = self.run_ablation(dimension, value, param)
            if checkpoint:
                detect_dir = self.run_detection(checkpoint, f"ablation_{dimension}_{value}")
                if detect_dir:
                    eval_result = self.run_evaluation(detect_dir, f"ablation_{dimension}_{value}")
                    if eval_result:
                        results["ablation"].append({
                            "dimension": dimension,
                            "value": value,
                            **eval_result
                        })
        
        # 5. 生成报告
        self.generate_report(results)
        
        elapsed = time.time() - start_time
        self.log("=" * 60)
        self.log(f"实验完成，总耗时: {elapsed/3600:.2f} 小时")
        self.log("=" * 60)


def main():
    runner = QuickExperimentRunner()
    runner.run_all()


if __name__ == "__main__":
    main()
