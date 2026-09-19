@echo off
setlocal enabledelayedexpansion
REM Activate the Conda environment
call conda activate look2hear_win

REM Make sure output folder exists
if not exist output (
    mkdir output
)

REM List of supported extensions
set EXTENSIONS=mp3 m4a wav aiff aif flac opus

REM First pass: count total matching files for a [current/total] indicator
set /a total=0
for %%e in (%EXTENSIONS%) do (
    for %%f in (input\*.%%e) do (
        set /a total+=1
    )
)

for /f %%s in ('powershell -NoProfile -Command "[int64](Get-Date -UFormat %%s)" ^< NUL') do set batch_start=%%s

REM Second pass: process each file
set /a counter=0
for %%e in (%EXTENSIONS%) do (
    for %%f in (input\*.%%e) do (
        set /a counter+=1
        REM Get the file name without extension
        set "filename=%%~nf"
        REM Skip files already processed in a previous run
        if exist "output\!filename!.wav" (
            echo [!counter!/!total!] Skipping %%f, output\!filename!.wav already exists
        ) else (
            echo [!counter!/!total!] Processing: %%f
            REM Run the Python inference script
            python inference2.py --in_wav="%%f" --out_wav="output\!filename!.wav"
        )
    )
)

for /f %%s in ('powershell -NoProfile -Command "[int64](Get-Date -UFormat %%s)" ^< NUL') do set batch_end=%%s
set /a elapsed=%batch_end% - %batch_start%

echo All files processed in !elapsed!s!
pause
