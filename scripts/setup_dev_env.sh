#!/bin/bash

# Set the local Python version to 3.10.12
pyenv local 3.10.12

# Create a virtual environment named .venv in the root folder
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install all dependencies listed in requirements.txt
pip install -r main/requirements.txt