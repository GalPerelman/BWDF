#!/bin/bash

dma=(0 1 2 3 4 5 6 7 8 9)
models=('prophet')
dates_idx=(3)
horizon=('short' 'long')
move_stats=1  # int to represent bool - 1 will include moving avg and moving std columns, 0 will not
decompose_target=1  # int to represent bool - 1 will decompose target to trend, seasonality and noise, 0 will not

for a in ${dma[@]}; do
	for b in ${models[@]}; do
	  for c in ${dates_idx[@]}; do
	    for d in ${horizon[@]}; do
          bash ./create_tmp_empty.sh "python ./wrapper.py
                                      --inflow_data_file Inflow_Data.xlsx
                                      --weather_data_file Weather_Data_2.xlsx
                                      --do experiment
                                      --search_params 1
                                      --dma_idx $a
                                      --model_name $b
                                      --dates_idx $c
                                      --horizon $d
                                      --norm_methods standard moving_stat fixed_window
                                      --target_lags 12 24
                                      --weather_lags 0 6
                                      --move_stats $move_stats
                                      --decompose_target $decompose_target
                                      --outliers_config_path outliers_config_w2.json
                                      --output_dir exp_output"

	    done
    done
  done
done