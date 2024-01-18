#!/bin/bash

dma=(0)
models=('xgb')
dates_idx=(0)
horizon=('short')

move_stats=1  # int to represent bool - 1 will include moving avg and moving std columns, 0 will not
decompose_target=1  # int to represent bool - 1 will decompose target to trend, seasonality and noise, 0 will not

for a in ${dma[@]}; do
	for b in ${models[@]}; do
	  for c in ${dates_idx[@]}; do
	    for d in ${horizon[@]}; do
          bash ./create_tmp_empty.sh "python ./wrapper.py
                                      --do experiment
                                      --search_params 1
                                      --dma_idx $a
                                      --model_name $b
                                      --dates_idx $c
                                      --horizon $d
                                      --norm_methods standard
                                      --target_lags 12
                                      --weather_lags 0
                                      --move_stats $move_stats
                                      --decompose_target $decompose_target
                                      --output_dir test_experiment"

	    done
    done
  done
done