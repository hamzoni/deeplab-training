#!/usr/bin/env bash

LABEL=$1
DIR=$(pwd)/dl/datasets/custom_models

echo ">>>>> MOVE ALL JSON FILES TO JSON FOLDER"
find images -name '*.json' -exec cp {} json \; | parallel pngquant

echo ">>>>> CONVERT JSON TO DATASETS"
python labelme/json_to_dataset.py -jf json -lb $LABEL -o json_out

echo ">>>>> CONVERT JPG IMAGES TO PNG IMAGES"
mogrify -format png images/*.jpg

echo ">>>>> MOVE ALL IMAGES TO IMAGE FOLDER"
find images -name '*.png' -exec cp {} $DIR/image \; | parallel pngquant

echo ">>>>> GT CONVERT"
python dl/gt_convert.py $(pwd)/json_out $(pwd)/gt

echo ">>>>> GENERATE TRAINING DATA"
python dl/train_data.py $DIR

echo ">>>>> COPY MASK IMAGES TO MASK FOLDER"
find gt -name '*.*' -exec cp {} $DIR/mask \; | parallel pngquant

echo ">>>>> RENAME MASK"
cd $DIR/mask
rename 's/\_gt.png/.png/' ./* #Modify suffix
rename 'y/A-Z/a-z/' * #All lowercase

# cd ../image
# rename 'y/A-Z/a-z/' * 
