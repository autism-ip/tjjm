#!/usr/bin/env python3
"""
优化训练脚本 - 12小时内完成
策略: 200CT + 30epochs + batch=16 + patch=100 + 缓存优化
"""

import os
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['OMP_NUM_THREADS'] = '1'


def create_optimized_dataset():
    """创建优化的小数据集（200CT）"""
    src_dir = Path('/root/autodl-tmp/data/raw/LUNA16')
    dst_dir = Path('/root/autodl-tmp/data/raw/LUNA16_200')
    
    if dst_dir.exists() and len(list(dst_dir.glob('*.mhd'))) == 200:
        print(f"✓ 数据集已存在: {dst_dir}")
        return str(dst_dir)
    
    dst_dir.mkdir(exist_ok=True)
    shutil.copy2(src_dir / 'annotations.csv', dst_dir / 'annotations.csv')
    
    mhd_files = sorted(src_dir.glob('*.mhd'))[:200]
    for i, mhd in enumerate(mhd_files):
        raw = mhd.with_suffix('.raw')
        shutil.copy2(mhd, dst_dir / mhd.name)
        shutil.copy2(raw, dst_dir / raw.name)
        if (i+1) % 50 == 0:
            print(f"已复制 {i+1}/200")
    
    print(f"✓ 子集创建完成: {dst_dir}")
    return str(dst_dir)


def main():
    import subprocess
    
    # 确保数据集存在
    data_dir = create_optimized_dataset()
    
    # 优化训练命令
    train_cmd = (
        "python scripts/train_autoencoder.py "
        f"data.dataset_dir={data_dir} "
        f"data.luna16_raw_dir={data_dir} "
        "data.batch_size=16 "
        "data.num_workers=8 "
        "model.encoder_name=swin_unetr "
        "model.encoder_pretrained=true "
        "model.freeze_encoder=true "
        "model.use_checkpoint=true "
        "loss.name=weighted_mse "
        "loss.weight_k=5.0 "
        "training.max_epochs=30 "
        "training.precision=16-mixed "
        "training.accelerator=gpu "
        "training.devices=1 "
        "training.seed=42 "
        "training.early_stopping.enabled=true "
        "training.early_stopping.patience=5 "
        "training.early_stopping.min_delta=0.001 "
        "training.checkpoint_dir=/root/tjjm/outputs/checkpoints/optimized_run"
    )
    
    print("=" * 60)
    print("优化训练配置:")
    print(f"  数据集: {data_dir} (200 CT)")
    print(f"  Epochs: 30")
    print(f"  Batch Size: 16")
    print(f"  Workers: 8")
    print(f"  早停: patience=5")
    print("=" * 60)
    
    # 执行训练
    log_file = "/root/tjjm/outputs/optimized_train.log"
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            train_cmd, shell=True,
            stdout=f, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT)
        )
        print(f"训练已启动，PID: {process.pid}")
        print(f"日志: {log_file}")
        process.wait()
    
    print("训练完成!")


if __name__ == "__main__":
    main()
