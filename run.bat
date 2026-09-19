@echo off
REM Activate the Conda environment
call conda activate look2hear_win

REM Run the batch driver: recurses input\, mirrors output\, skips existing
REM files, and calls inference2.py once per file. See run_batch.py for why
REM this bookkeeping lives in Python rather than in this script.
python run_batch.py

pause
