#!/bin/bash
set -x

PYTHON_LIB_FOLDER=/home/bishal/Programming/.venv/lib

export LIBSUMO_AS_TRACI=1
export SUMO_HOME="$PYTHON_LIB_FOLDER/python3.11/site-packages/sumo"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "Set the SUMO_HOME environment variable to $SUMO_HOME"

set +x

