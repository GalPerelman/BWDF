@echo off
REM Step 1: Freeze the environment and save to a temporary file
pip freeze > temp_requirements.txt

REM Step 2: Process the temporary file to remove versions and save to the final file
(for /f "tokens=1 delims==" %%a in (temp_requirements.txt) do echo %%a) > requirements_no_versions.txt

REM Cleanup the temporary file
del temp_requirements.txt