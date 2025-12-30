#!/bin/bash

# Training Script for Physio-SynBrain
# Usage: ./scripts/run_train.sh [subject_id] [batch_size]

SUB=${1:-1}
BATCH_SIZE=${2:-4} # Reduced batch size due to large memory footprint of 31k voxels * 48 time steps
DATA_ROOT="data/NSD"
OUTPUT_DIR="checkpoints/sub${SUB}"

echo "Starting training for Subject ${SUB}..."
echo "Data Root: ${DATA_ROOT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Batch Size: ${BATCH_SIZE}"

# Using accelerate for mixed precision and easy multi-gpu support if configured
accelerate launch --multi_gpu -m src.physio_synbrain.train \
    --sub $SUB \
    --data_root $DATA_ROOT \
    --batch_size $BATCH_SIZE \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir $OUTPUT_DIR
