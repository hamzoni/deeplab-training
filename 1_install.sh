#!/usr/bin/env bash

sudo apt install pngquant -y
sudo apt install parallel -y
sudo apt-get install imagemagick -y &&

echo '>>>> Installing dependencies'
python labelme/setup.py build
python labelme/setup.py install

# sudo -s
cd models-master/research
python setup.py build
python setup.py install
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ../..
