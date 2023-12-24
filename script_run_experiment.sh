#!/bin/bash

dma=(0 1 2 3 4 5 6 7 8 9)
models=('xgb' 'prophet' 'lstm' 'multi_series')
dates_idx=(0 1 2 3)
horizon=('short' 'long')
norm_method=('standard' 'min_max' 'robust' 'power' 'quantile' 'moving_stat' 'fixed_window')

for a in ${dma[@]}; do
	for b in ${models[@]}; do
	  for c in ${dates_idx[@]}; do
	    for d in ${horizon[@]}; do
	      for e in ${norm_method[@]}; do
		      bash ./create_tmp_empty.sh "python ./wrapper.py --do experiment --dma_idx $a --model_name $b --dates_idx $c --horizon $d --norm_method $e --output_dir exp_output"
		    done
	    done
    done
  done
done