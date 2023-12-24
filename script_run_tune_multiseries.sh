#!/bin/bash

dma=(5 6 7 8 9)

for a in ${dma[@]}; do
  bash ./create_tmp_empty.sh "python ./wrapper.py --do multi_series_tune --dma_idx $a"
done