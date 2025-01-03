#!/bin/bash

# Download the data
wget -nc https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_1024_10K/data_000000.tar -P data

# untar the data
tar -xvf data/data_000000.tar -C data
