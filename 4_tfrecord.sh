#!/usr/bin/env bash

DIR=$(pwd)/dl/datasets/custom_models

echo '>>>>> Working on directory: ' $DIR

python3 dl/datasets/build_voc2012_data.py \
      --image_folder="$DIR/image" \
      --semantic_segmentation_folder="$DIR/mask" \
      --list_folder="$DIR/index" \
      --image_format="png" \
      --output_dir="$DIR/tfrecord"
