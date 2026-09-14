#!/bin/bash
echo "Starting Live PCA Engine with High-Performance Dedicated GPU..."

# Force dedicated GPU on Linux hybrid graphics (NVIDIA Optimus / AMD PRIME)
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export DRI_PRIME=1

# Add current directory to PYTHONPATH
export PYTHONPATH=.

# Run the app
python main.py
