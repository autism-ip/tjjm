#!/usr/bin/env python3
"""
小数据集快速实验 - 使用前10个CT文件
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

os.environ['OMP_NUM_THREADS'] = '1'


class SmallExperimentRunner:
    """小数据集实验执行器"""
    
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
    
    def create_small_dataset(self):
        """创建小数据集（前10个CT文件）"""
        small_dir = Path(self.DATA_ROOT).parent / "LUNA16_small"
        small_dir.mkdir(exist_ok=True)
        
        # 复制annotations.csv
        import shutil
        src_ann = Path(self.DATA_ROOT) / "annotations.csv"
        dst_ann = small_dir / "annotations.csv"
        if not dst_ann.exists():
            shutil.copy2(src_ann, dst_ann)
        
        # 复制前10个CT文件
        mhd_files = sorted(Path(self.DATA_ROOT).glob("*.mhd"))[:10]
        for mhd in mhd_files:
            raw = mhd.with_suffix(".raw")
            dst_mhd = small_dir / mhd.name
            dst_raw = small_dir / raw.name
            if not dst_mhd.exists():
                shutil.copy2(mhd, dst_mhd)
            if not dst_raw.exists():
                shutil.copy2(raw, dst_raw)
        
        self.log(f"✓ 小数据集创建完成: {small_dir} ({len(mhd_files)} 个CT)")
        return str(small_dir)
    
    def run_command(self, cmd: str, description: str = "") -> bool:
        if description:
            self.log(f"执行: {description}")
        self.log(f"命令: {cmd[:150]}...")
        
        try:
            result = subprocess.run(
                cmd, shell=True, check=True,
                capture_output=True, text=True,
                cwd=str(PROJECT_ROOT),
                timeout=1800  # 30分钟超时
            )
            self.log("✓ 成功")
            return True
        except subprocess.TimeoutExpired:
            self.log("✗ 超时")
            return False
        except subprocess.CalledProcessError as e:
            self.log(f"✗ 失败: {e.stderr[:300] if e.stderr else 'Unknown error'}")
            return False
    
    def run_smoke_test(self, data_dir: str) -> bool:
        """Smoke Test"""
        self.log("=" * 60)
        self.log("Smoke Test")
        self.log("=" * 60)
        
        train_cmd = (
            "python scripts/train_autoencoder.py "
            f"data.dataset_dir={data_dir} "
            f"data.luna16_raw_dir={data_dir} "
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
        
        checkpoint = Path(self.CHECKPOINT_BASE) / "smoke_test" / "final.ckpt"
        if not checkpoint.exists():
            self.log("Checkpoint 未生成")
            return False
        
        self.log(f"✓ Smoke Test 通过: {checkpoint}")
        return True
    
    def run_baseline(self, data_dir: str) -> Path:
        """基线训练"""
        self.log("=" * 60)
        self.log("基线训练")
        self.log("=" * 60)
        
        checkpoint_dir = Path(self.CHECKPOINT_BASE) / "baseline"
        
        train_cmd = (
            "python scripts/train_autoencoder.py "
            f"data.dataset_dir={data_dir} "
            f"data.luna16_raw_dir={data_dir} "
            "data.batch_size=4 "
            "data.num_workers=2 "
            "model.encoder_name=swin_unetr "
            "model.encoder_pretrained=true "
            "model.freeze_encoder=true "
            "loss.name=weighted_mse "
            "training.max_epochs=100 "
            "training.precision=16-mixed "
            "training.accelerator=gpu "
            "training.devices=1 "
            "training.seed=42 "
            "training.early_stopping.enabled=true "
            "training.early_stopping.patience=10 "
            f"training.checkpoint_dir={checkpoint_dir}"
        )
        
        if self.run_command(train_cmd, "基线训练"):
            checkpoint = checkpoint_dir / "final.ckpt"
            if checkpoint.exists():
                self.log(f"✓ 基线训练完成: {checkpoint}")
                return checkpoint
        return None
    
    def run_ablation(self, data_dir: str, dimension: str, value: str, param: str) -> Path:
        """消融实验"""
        self.log("=" * 60)
        self.log(f"消融: {dimension}={value}")
        self.log("=" * 60)
        
        checkpoint_dir = Path(self.CHECKPOINT_BASE) / f"ablation_{dimension}_{value}"
        
        train_cmd = (
            "python scripts/train_autoencoder.py "
            f"data.dataset_dir={data_dir} "
            f"data.luna16_raw_dir={data_dir} "
            "data.batch_size=4 "
            "data.num_workers=2 "
            f"{param}={value} "
            "loss.name=weighted_mse "
            "training.max_epochs=100 "
            "training.precision=16-mixed "
            "training.accelerator=gpu "
            "training.devices=1 "
            "training.seed=42 "
            "training.early_stopping.enabled=true "
            "training.early_stopping.patience=10 "
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
        self.log("生成报告")
        self.log("=" * 60)
        
        report = []
        report.append("# Lung-Diffusion-Anomaly 快速实验报告")
        report.append("")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**数据集**: 小数据集（10个CT文件）")
        report.append("")
        
        if "ablation" in results:
            report.append("## 消融实验结果")
            report.append("")
            report.append("| 维度 | 值 | Checkpoint |")
            report.append("|------|-----|------------|")
            for item in results["ablation"]:
                report.append(f"| {item['dimension']} | {item['value']} | {item['checkpoint']} |")
            report.append("")
        
        report_file = Path(self.OUTPUT_BASE) / f"quick_report_{self.run_id}.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report))
        
        self.log(f"✓ 报告已生成: {report_file}")
    
    def run_all(self):
        """执行所有实验"""
        self.log("=" * 60)
        self.log("开始快速实验")
        self.log("=" * 60)
        
        start_time = time.time()
        results = {"ablation": []}
        
        # 创建小数据集
        data_dir = self.create_small_dataset()
        
        # 1. Smoke Test
        if not self.run_smoke_test(data_dir):
            self.log("Smoke Test 失败")
            return
        
        # 2. 基线训练
        baseline_checkpoint = self.run_baseline(data_dir)
        if not baseline_checkpoint:
            self.log("基线训练失败")
            return
        
        # 3. 消融实验
        ablation_configs = [
            ("loss", "mse", "loss.name"),
            ("encoder", "resnet", "model.encoder_name"),
        ]
        
        for dimension, value, param in ablation_configs:
            checkpoint = self.run_ablation(data_dir, dimension, value, param)
            if checkpoint:
                results["ablation"].append({
                    "dimension": dimension,
                    "value": value,
                    "checkpoint": str(checkpoint),
                })
        
        # 4. 生成报告
        self.generate_report(results)
        
        elapsed = time.time() - start_time
        self.log("=" * 60)
        self.log(f"实验完成，总耗时: {elapsed/60:.2f} 分钟")
        self.log("=" * 60)


def main():
    runner = SmallExperimentRunner()
    runner.run_all()


if __name__ == "__main__":
    main()
