#!/bin/bash

echo "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

echo "Installing Python and dependencies..."
sudo apt-get install -y python3-pip python3-venv libgl1-mesa-glx

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Python libraries..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Active virtual environment with 'source venv/bin/activate'"
