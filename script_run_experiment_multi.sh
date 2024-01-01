#!/bin/bash

dma=(0 1 2 3 4 5 6 7 8 9)
models=('multi')
dates_idx=(0 1)
horizon=('short' 'long')
target_lags_min=12
target_lags_step=12
target_lags_steps=0
weather_lags_min=0
weather_lags_step=0
weather_lags_steps=1
move_stats=0  # int to represent bool - 1 will include moving avg and moving std columns, 0 will not
decompose_target=0  # int to represent bool - 1 will decompose target to trend, seasonality and noise, 0 will not

for a in ${dma[@]}; do
	for b in ${models[@]}; do
	  for c in ${dates_idx[@]}; do
	    for d in ${horizon[@]}; do
          bash ./create_tmp_empty.sh "python ./wrapper.py
                                      --do experiment
                                      --dma_idx $a
                                      --model_name $b
                                      --dates_idx $c
                                      --horizon $d
                                      --target_lags_min $target_lags_min
                                      --target_lags_step $target_lags_step
                                      --target_lags_steps $target_lags_steps
                                      --weather_lags_min $weather_lags_min
                                      --weather_lags_step $weather_lags_step
                                      --weather_lags_steps $weather_lags_steps
                                      --move_stats $move_stats
                                      --decompose_target $decompose_target
                                      --output_dir exp_output"

	    done
    done
  done
done