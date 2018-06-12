#!/bin/bash

# Download the parent model to directly use it for online training
wget https://data.vision.ee.ethz.ch/kmaninis/share/OSVOS/Downloads/models/pth_parent_model.zip
unzip pth_parent_model.zip
rm pth_parent_model.zip
