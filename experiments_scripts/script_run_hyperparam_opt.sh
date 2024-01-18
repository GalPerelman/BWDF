#!/bin/bash

dma=(0 1 2 3 4 5 6 7 8 9)
models=('lstm')
dates_idx=(0)
norm_method=('fixed_window')
target_lags=(12)

for a in ${dma[@]}; do
  for b in ${models[@]}; do
	  for c in ${dates_idx[@]}; do
	    for d in ${norm_method[@]}; do
	      for e in ${target_lags[@]}; do
          bash ./create_tmp_empty.sh "python ./wrapper.py --do hyperparam_opt --dma_idx $a --model_name $b
                                      --dates_idx $c --norm_method $d --target_lags $e"
        done
      done
    done
  done
done