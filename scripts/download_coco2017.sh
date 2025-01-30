#!/usr/bin/env bash
#
# Script: download_coco2017_val.sh
# Description: Download MS COCO 2017 val images (5k) and the annotation files (which include captions).
# Usage: ./download_coco2017_val.sh

# 1. Create a directory to hold COCO 2017 val data and move into it
mkdir -p coco2017_val
cd coco2017_val

# 2. Download the val images
echo "Downloading COCO 2017 Val images..."
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip

# 3. Download the train/val annotations, which include val caption annotations
echo "Downloading COCO 2017 Train/Val Annotations..."
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

# The relevant val caption file is now in annotations/captions_val2017.json
echo "Done! val2017 images and annotations downloaded in $(pwd)."
echo "Use 'annotations/captions_val2017.json' for the ground-truth captions."