call "venv\Scripts\activate.bat"
python main.py --inflow_data_file Inflow_Data_2.xlsx --weather_data_file Weather_Data_2.xlsx --predict test --models_config models_config_w2.json --test_name w2 --plot true
pause