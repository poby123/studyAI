#!/bin/bash
curl -L -o data/fashionmnist.zip https://www.kaggle.com/api/v1/datasets/download/zalando-research/fashionmnist
unzip data/fashionmnist.zip -d data/fashionmnist
rm data/fashionmnist.zip