#!/bin/bash

dma=(0)
models=('patch')
dates_idx=(3)
horizon=('short', 'long')

for a in ${dma[@]}; do
	for b in ${models[@]}; do
	  for c in ${dates_idx[@]}; do
	    for d in ${horizon[@]}; do
          bash ./create_tmp_empty.sh "python ./wrapper.py
                                      --do nixtla
                                      --search_params 1
                                      --dma_idx $a
                                      --model_name $b
                                      --dates_idx $c
                                      --horizon $d
                                      --norm_methods standard fixed_window
                                      --target_lags 24
                                      --weather_lags 0
                                      --output_dir exp_output"

	    done
    done
  done
done