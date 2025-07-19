#!/bin/bash

# Base directories
FEATURES_DIR="../features"

# ImageNet 
python compute_trust_score.py \
    --dataset_name "imagenet" \
    --model_name "resnet50" \
    --features_dir $FEATURES_DIR

# Places365 
python compute_trust_score.py \
    --dataset_name "places" \
    --model_name "resnet152_places" \
    --features_dir $FEATURES_DIR

# Fitzpatrick17k 
python compute_trust_score.py \
    --dataset_name "fitzpatrick17k" \
    --model_name "resnet18_fitzpatrick17k" \
    --features_dir $FEATURES_DIR

# ImageNet-LT 
python compute_trust_score.py \
    --dataset_name "imagenet_lt" \
    --model_name "resnext50_imagenet_lt" \
    --features_dir $FEATURES_DIR

# Places365-LT 
python compute_trust_score.py \
    --dataset_name "places_lt" \
    --model_name "resnet152_places_lt" \
    --features_dir $FEATURES_DIR 