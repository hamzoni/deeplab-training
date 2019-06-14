#!/usr/bin/env bash


INITIAL_CHECKPOINT=$(pwd)/dl/datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/model.ckpt
LOG_DIR=$(pwd)/dl/datasets/custom_models/logs
DATASET_DIR=$(pwd)/dl/datasets/custom_models/tfrecord


# TRAIN FRESH
rm -rf $(pwd)/dl/datasets/custom_models/logs/*
python dl/train.py \
    --logtostderr \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=513,513 \
    --train_batch_size=8 \
    --training_number_of_steps=20000 \
    --fine_tune_batch_norm=false \
    --train_logdir=$LOG_DIR \
    --dataset="taquy" \
    --num_clones=4 \
    --ps_tasks=1 \
    --dataset_dir=$DATASET_DIR 2>&1

# RETRAIN MODEL
# python dl/train.py \
#     --logtostderr \
#     --train_split="train" \
#     --model_variant="xception_65" \
#     --atrous_rates=6 \
#     --atrous_rates=12 \
#     --atrous_rates=18 \
#     --output_stride=16 \
#     --decoder_output_stride=4 \
#     --train_crop_size=513,513 \
#     --train_batch_size=1 \
#     --training_number_of_steps=50 \
#     --fine_tune_batch_norm=false \
#     --tf_initial_checkpoint=$INITIAL_CHECKPOINT \
#     --train_logdir=$LOG_DIR \
#     --dataset="taquy" \
#     --dataset_dir=$DATASET_DIR 2>&1

