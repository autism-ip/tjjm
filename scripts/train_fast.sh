#!/bin/bash
# 激进优化训练 - 12小时内完成
# 策略: 50CT + stride=128 + batch=16 + 30epochs

export OMP_NUM_THREADS=1
cd /root/tjjm

DATA_DIR="/root/autodl-tmp/data/raw/LUNA16_50"
CHECKPOINT_DIR="/root/tjjm/outputs/checkpoints/fast_run"
LOG_FILE="/root/tjjm/outputs/fast_train.log"

echo "========================================" > $LOG_FILE
echo "激进优化训练" | tee -a $LOG_FILE
echo "  数据集: $DATA_DIR (50 CT)" | tee -a $LOG_FILE
echo "  Epochs: 30" | tee -a $LOG_FILE
echo "  Batch Size: 16" | tee -a $LOG_FILE
echo "  Stride: 128" | tee -a $LOG_FILE
echo "  早停: patience=5" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

exec python scripts/train_autoencoder.py \
    data.dataset_dir=$DATA_DIR \
    data.luna16_raw_dir=$DATA_DIR \
    data.batch_size=16 \
    data.num_workers=8 \
    data.patch_stride=128 \
    model.encoder_name=swin_unetr \
    model.encoder_pretrained=true \
    model.freeze_encoder=true \
    model.use_checkpoint=true \
    loss.name=weighted_mse \
    loss.weight_k=5.0 \
    training.max_epochs=30 \
    training.precision=16-mixed \
    training.accelerator=gpu \
    training.devices=1 \
    training.seed=42 \
    training.early_stopping.enabled=true \
    training.early_stopping.patience=5 \
    training.early_stopping.min_delta=0.001 \
    training.checkpoint_dir=$CHECKPOINT_DIR \
    >> $LOG_FILE 2>&1
