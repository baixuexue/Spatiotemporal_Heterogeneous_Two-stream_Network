#!/usr/bin/env bash

TOOLS=lib/caffe-action/build/install/bin
LOG_FILE=logs/spatiotemporal_heterogeneous_temporal_split1.log
N_GPU=8
MPI_BIN_DIR=/usr/bin/


echo "logging to ${LOG_FILE}"

${MPI_BIN_DIR}mpirun -np $N_GPU \
$TOOLS/caffe train --solver=spatiotemporal_heterogeneous/temporal_bn_inception_solver.prototxt \
	--weights=pretrainmodels/bn_inception_init_flow.caffemodel 2>&1 | tee ${LOG_FILE}