#!/bin/bash
# 优化训练启动脚本
# 策略: 200CT + 30epochs + batch=16 + 早停

export OMP_NUM_THREADS=1
cd /root/tjjm

DATA_DIR="/root/autodl-tmp/data/raw/LUNA16_200"
CHECKPOINT_DIR="/root/tjjm/outputs/checkpoints/optimized_run"
LOG_FILE="/root/tjjm/outputs/optimized_train.log"

echo "========================================" | tee -a $LOG_FILE
echo "优化训练启动" | tee -a $LOG_FILE
echo "  数据集: $DATA_DIR (200 CT)" | tee -a $LOG_FILE
echo "  Epochs: 30" | tee -a $LOG_FILE
echo "  Batch Size: 16" | tee -a $LOG_FILE
echo "  早停: patience=5" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

exec python scripts/train_autoencoder.py \
    data.dataset_dir=$DATA_DIR \
    data.luna16_raw_dir=$DATA_DIR \
    data.batch_size=16 \
    data.num_workers=8 \
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
