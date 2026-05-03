#!/bin/bash
# 后台训练脚本 - 彻底分离进程
export OMP_NUM_THREADS=1
cd /root/tjjm
exec python scripts/train_autoencoder.py \
    data.dataset_dir=/root/autodl-tmp/data/raw/LUNA16_small \
    data.luna16_raw_dir=/root/autodl-tmp/data/raw/LUNA16_small \
    data.batch_size=4 \
    data.num_workers=2 \
    training.max_epochs=10 \
    training.accelerator=gpu \
    training.checkpoint_dir=/root/tjjm/outputs/checkpoints/baseline \
    >> /root/tjjm/outputs/baseline_train.log 2>&1
