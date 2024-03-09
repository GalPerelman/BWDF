#!/bin/bash

dma=(0 1 2 3 4 5 6 7 8 9)
horizon=('short' 'long')

for a in ${dma[@]}; do
	for b in ${horizon[@]}; do
      bash ./create_tmp_empty.sh "python ./wrapper.py
                                      --inflow_data_file Inflow_Data_4.xlsx
                                      --weather_data_file Weather_Data_4.xlsx
                                      --do cv
                                      --cv_start_year 2023
                                      --cv_start_month 2
                                      --cv_start_day 13
                                      --search_params 0
                                      --dma_idx $a
                                      --horizon $b
                                      --outliers_config_path outliers_config_w4.json
                                      --cv_candidates_path experiments_analysis/candidates_${b}_w4.json
                                      --output_dir cv_w4"

	done
done