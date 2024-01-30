# BWDF
This repository contains the code for the Battle of Water Demand Forecast submission

### Usage
##### Setup a Python virtual environment
First, initiate a Python virtual environment
The code was developed with Python 3.10.11
The code was tested with Python 3.8.10

Open a terminal window in the project directory
Create new venv by running the command `python3 -m venv <venv name>`</br>
Activate the venv by `source venv/bin/activate` (for mac) or `venv/Scripts/activate.bat` (for windows)</br>

##### Install dependencies
Once the virtual environment is setup and activated,
Install the dependencies by running the following command:
`pip install -r requirements.txt`</br>

The `requirement.txt` file includes the packages with the specific versions used to develop the code with Windows OS
It was found that for other operation systems, packages versions might be different
An alternative dependencies installation can be done by running:
`pip install -r requirements_no_versions.txt`</br>
This command will install the latest version of each package

##### Run the code
To run the code and generate a forecast for all the DMAs run `main.py`
Example usage to run the first test period
`python main.py --predict test --models_config models_config_w1.json --test_name w1 --plot true`

##### Arguments
`predict`          Must be one of `test` or `experiment`</br>
`models_config`    Path to a json file that defines the models configuration</br>
`test_name`        Pass only if `predict` argument is `test`. The test period to predict, must be one of `w1`, `w2`, `w3`, `w4`</br>
`experiment_idx`   Pass only if `predict` argument is `experiment`. The experiment index</br>
`plot`             true for plot the forecast, false otherwise. default is true</br>
`export`           true for export forecast to csv, false otherwise. default is true</br>
