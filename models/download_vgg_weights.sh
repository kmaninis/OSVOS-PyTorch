#!/bin/bash

# Download the parent model to directly use it for online training
wget https://data.vision.ee.ethz.ch/kmaninis/share/OSVOS/Downloads/models/vgg_mat.zip
unzip vgg_mat.zip
rm vgg_mat.zip
