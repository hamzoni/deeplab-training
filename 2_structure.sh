#!/usr/bin/env bash

echo '>>>> Create folder structure'
cd dl/datasets
rm -rf custom_models
mkdir custom_models
cd custom_models
mkdir exp image index mask tfrecord
cd ../../..
mkdir images
mkdir json
mkdir json_out
mkdir gt

echo '>>>> Done'
