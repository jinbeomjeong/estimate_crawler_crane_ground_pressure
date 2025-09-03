#!/bin/bash

source /home/pi/miniconda3/bin/activate

conda activate tensorflow_cpu_219_python_310

python /home/pi/workspace/test/modbus_server.py
