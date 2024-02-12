#!/bin/bash

dma=(0 1 2 3 4 5 6 7 8 9)
horizon=('short' 'long')

for a in ${dma[@]}; do
	for b in ${horizon[@]}; do
      bash ./create_tmp_empty.sh "python ./wrapper.py
                                      --inflow_data_file Inflow_Data.xlsx
                                      --weather_data_file Weather_Data_2.xlsx
                                      --do cv
                                      --cv_start_year 2022
                                      --cv_start_month 10
                                      --cv_start_day 3
                                      --search_params 0
                                      --dma_idx $a
                                      --horizon $b
                                      --outliers_config_path outliers_config_w2.json
                                      --cv_candidates_path experiments_analysis/candidates_${b}_w2.json
                                      --output_dir cv_w2"

	done
done