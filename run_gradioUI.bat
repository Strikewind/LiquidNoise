@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

set CONDA_PATH=
for %%p in (
    "%USERPROFILE%\anaconda3\envs\fyp\python.exe"
    "%USERPROFILE%\miniconda3\envs\fyp\python.exe"
    "%PROGRAMFILES%\anaconda3\envs\fyp\python.exe"
    "%PROGRAMFILES%\miniconda3\envs\fyp\python.exe"
    "%PROGRAMFILES(x86)%\anaconda3\envs\fyp\python.exe"
    "%PROGRAMFILES(x86)%\miniconda3\envs\fyp\python.exe"
    "%PROGRAMDATA%\Anaconda3\envs\fyp\python.exe"
    "%PROGRAMDATA%\Miniconda3\envs\fyp\python.exe"
) do (
    if exist %%p set CONDA_PATH=%%p
)

if "%CONDA_PATH%"=="" (
    echo Conda executable not found. Please ensure Anaconda or Miniconda is installed.
)

cd /d "%~dp0"

"%CONDA_PATH%" gradioUI.py

ENDLOCAL

pause