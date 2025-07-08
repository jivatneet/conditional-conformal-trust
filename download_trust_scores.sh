#!/bin/bash

# Install gdown if not already installed
pip install gdown

# Download the zip file using gdown
echo "Downloading trust scores..."
FILE_ID="1csaSMPNfm9ItkvrGkqcoHeqbzNIkLZrb"
gdown $FILE_ID -O trust_scores.zip

# Unzip the file into current directory
echo "Extracting trust scores..."
unzip -o trust_scores.zip

# Remove the zip file
echo "Cleaning up..."
rm trust_scores.zip

echo "Done! Trust scores have been downloaded and extracted." 