#!/bin/bash

source /home/pi/miniconda3/bin/activate

conda activate onnx_runtime_python_311

python /home/pi/workspace/estimate_crawler_crane_ground_pressure/test.py
