@echo off
for /f "tokens=1 delims==" %%a in ('findstr /v "^#"') do echo %%a