# BWDF
This repository contains the code for the Battle of Water Demand Forecast submission

### Usage
##### Setup a Python virtual environment
First, initiate a Python virtual environment
The code was developed with Python 3.10.11
The code was tested with Python 3.8.10
Other Python version above 3.6 should be fine but were not tested

Open a terminal window in the project directory
Create new venv by running the command:</br>
`python -m venv <venv name>`</br>
Activate the venv by:</br>
`source venv/bin/activate` (for mac) or `venv\Scripts\activate.bat` (for windows)</br>

##### Install dependencies
Once the virtual environment is setup and activated,
Install the dependencies by running the following command:
`pip install -r requirements_no_versions.txt`</br>
Due to the large number of dependencies, the installation might take 10-15 minutes

##### Run the code
To run the code and generate a forecast for all the DMAs `main.py` should be run</br>
Example usage to run the first test period</br>
`python main.py --inflow_data_file Inflow_Data_1.xlsx --weather_data_file Weather_Data_1.xlsx --predict test --models_config models_config_w1.json --test_name w1 --plot true`
`python main.py --inflow_data_file Inflow_Data_2.xlsx --weather_data_file Weather_Data_2.xlsx --predict test --models_config models_config_w2.json --test_name w2 --plot true`
`python main.py --inflow_data_file Inflow_Data_3.xlsx --weather_data_file Weather_Data_3.xlsx --predict test --models_config models_config_w3.json --test_name w3 --plot true`

Alternative way to run predictions:</br>
The repository also contains `.bat` files for quick run.
These files assume the virtual environment was created and it is named `venv`
To run the first test (w1) double-click the file `predict_w1.bat`
To run the second test (w2) double-click the file `predict_w2.bat`
To run the third test (w3) double-click the file `predict_w3.bat`

The run time is expected to be 5-10 minutes
During the run the code will print to console according to its progress:</br>
Predicting DMA A</br>
Predicting DMA B</br>
...</br>

##### Arguments
`inflow_data_file` - Historic inflow data file, name only, not file path. file is located in `resources` directory</br></br>
`weather_data_file` - Weather data file, name only, not file path. file is located in `resources` directory</br>
`predict` - Must be one of `test` or `experiment`</br>
`models_config` - Path to a json file that defines the models configuration</br>
`test_name` - Pass only if `predict` argument is `test`. The test period to predict, must be one of `w1`, `w2`, `w3`, `w4`</br>
`experiment_idx` - Pass only if `predict` argument is `experiment`. The experiment index</br>
`plot` - "true" for plot the forecast, false otherwise. default is true</br>
`export` - "true" for export forecast to csv, false otherwise. default is true</br>


### Output
The code generates two outputs:
1) A CSV file with the forecast - should be manually copy and paste to the BWDF template.</br>
The file is written to the local folder as `forecast-test-<test_name>.csv` or `forecast-experiment-<experiment_idx>.csv` 
2) Forecast plot - Written to the local folder as `forecast-test-<test_name>.png` or `forecast-experiment-<experiment_idx>.png`