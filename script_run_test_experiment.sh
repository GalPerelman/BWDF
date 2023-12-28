#!/bin/bash

dma=(0 1)
models=('xgb')
dates_idx=(0)
horizon=('short' 'long')

for a in ${dma[@]}; do
	for b in ${models[@]}; do
	  for c in ${dates_idx[@]}; do
	    for d in ${horizon[@]}; do
          bash ./create_tmp_empty.sh "python ./wrapper.py
                                      --do test_experiment
                                      --dma_idx $a
                                      --model_name $b
                                      --dates_idx $c
                                      --horizon $d
                                      --output_dir exp_output"

	    done
    done
  done
done