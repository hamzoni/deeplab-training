#!/usr/bin/env bash

DIR=$(pwd)/dl/datasets/custom_models/logs

python3 dl/export_model.py \
  --logtostderr \
  --checkpoint_path="$DIR/model.ckpt-$1" \
  --export_path="$DIR/frozen_inference_graph.pb" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=2 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0

