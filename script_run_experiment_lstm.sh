#!/bin/bash

dma=(0 1 2 3 4 5 6 7 8 9)
models=('lstm')
dates_idx=(3)
horizon=('short' 'long')
move_stats=1  # int to represent bool - 1 will include moving avg and moving std columns, 0 will not
decompose_target=1  # int to represent bool - 1 will decompose target to trend, seasonality and noise, 0 will not

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
                                      --norm_methods standard moving_stat fixed_window
                                      --target_lags 12 24
                                      --weather_lags 0 6
                                      --move_stats $move_stats
                                      --decompose_target $decompose_target
                                      --output_dir exp_output"

	    done
    done
  done
done