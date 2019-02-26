#!/usr/bin/env bash

TOOLS=lib/caffe-action/build/install/bin
LOG_FILE=logs/isomorphism_res152_spatial_split1.log
N_GPU=8
MPI_BIN_DIR=/usr/bin/


echo "logging to ${LOG_FILE}"

${MPI_BIN_DIR}mpirun -np $N_GPU \
$TOOLS/caffe train --solver=isomorphism_res152/spatial_ResNet_152_solver.prototxt \
	--weights=pretrainmodels/ResNet-152-model_init_rgb.caffemodel 2>&1 | tee ${LOG_FILE}


