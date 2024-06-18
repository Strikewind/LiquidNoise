#!/bin/bash

CONDA_PATHS=(
    "$HOME/anaconda3/envs/fyp/bin/python"
    "$HOME/miniconda3/envs/fyp/bin/python"
    "/usr/local/anaconda3/envs/fyp/bin/python"
    "/usr/local/miniconda3/envs/fyp/bin/python"
    "/usr/anaconda3/envs/fyp/bin/python"
    "/usr/miniconda3/envs/fyp/bin/python"
    "/opt/anaconda3/envs/fyp/bin/python"
    "/opt/miniconda3/envs/fyp/bin/python"
)

CONDA_PATH=""

for path in "${CONDA_PATHS[@]}"; do
    if [ -f "$path" ]; then
        CONDA_PATH="$path"
        break
    fi
done

if [ -z "$CONDA_PATH" ]; then
    echo "Conda executable not found. Please ensure Anaconda or Miniconda is installed."
    exit 1
fi

cd "$(dirname "$0")"

"$CONDA_PATH" gradioUI.py