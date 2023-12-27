#!/bin/bash

dma=(0 1 2 3 4 5 6 7 8 9)
models=('xgb' 'prophet' 'lstm' 'multi_series')
dates_idx=(0 1 2 3)
horizon=('long')
norm_method=('standard' 'min_max' 'moving_stat' 'fixed_window')
target_lags_min=(0)
target_lags_step=(12)
target_lags_steps=(2)
weather_lags_min=(0)
weather_lags_step=(6)
weather_lags_steps=(2)
move_stats=(1)  # int to represent bool - 1 will include moving avg and moving std columns, 0 will not
decompose_target=(1)  # int to represent bool - 1 will decompose target to trend, seasonality and noise, 0 will not


for a in ${dma[@]}; do
	for b in ${models[@]}; do
	  for c in ${dates_idx[@]}; do
	    for d in ${horizon[@]}; do
	      for e in ${norm_method[@]}; do
	        for f in ${target_lags_min[@]}; do
	          for g in ${target_lags_step[@]}; do
	            for h in ${target_lags_steps[@]}; do
	              for i in ${weather_lags_min[@]}; do
	                for j in ${weather_lags_step[@]}; do
	                  for k in ${weather_lags_steps[@]}; do
	                    for l in ${move_stats[@]}; do
	                      for m in ${decompose_target[@]}; do
                            bash ./create_tmp_empty.sh "python ./wrapper.py
                                                        --do experiment
                                                        --dma_idx $a
                                                        --model_name $b
                                                        --dates_idx $c
                                                        --horizon $d
                                                        --norm_method $e
                                                        --target_lags_min $f
                                                        --target_lags_step $g
                                                        --target_lags_steps $h
                                                        --weather_lags_min $i
                                                        --weather_lags_step $j
                                                        --weather_lags_steps $k
                                                        --move_stats $l
                                                        --decompose_target $m
                                                        --output_dir exp_output"
                        done
                      done
                    done
                  done
                done
              done
            done
          done
		    done
	    done
    done
  done
done