#!/bin/bash

dma=(0 1 2 3 4 5 6 7 8 9)
horizon=('short' 'long')

for a in ${dma[@]}; do
	for b in ${horizon[@]}; do
      bash ./create_tmp_empty.sh "python ./wrapper.py
                                      --do cv
                                      --search_params 0
                                      --dma_idx $a
                                      --horizon $b
                                      --cv_candidates_path experiments_analysis/candidates_${b}_v1.json"

	done
done