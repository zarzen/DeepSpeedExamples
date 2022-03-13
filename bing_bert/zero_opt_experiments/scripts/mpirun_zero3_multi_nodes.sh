#!/bin/bash

full_path=$(realpath $0)

base_dir=$(dirname $full_path)


datetime=$(date +"%Y-%m-%d_%T")
echo "base dir ${base_dir}"
CONFIG_DIR=${base_dir}/../configs
HOST_FILE=${base_dir}/multi_nodes_hosts

CONFIG=${CONFIG_DIR}/1.5B_cgFalse_mbs8_loss_test.json

# locate the training script
TRAIN_SCRIPT_PATH=${base_dir}/../../deepspeed_train.py

# Where should we save checkpoints and tensorboard events?
JOB_NAME=zero3_multi_nodes_${datetime}
BASE_OUTPUT=${base_dir}/../outputs/${JOB_NAME}

mkdir -p $BASE_OUTPUT

DATA_PREFIX=/home/$USER/small-data-backup

MAX_STEPS=10

OUTPUT_DIR=$BASE_OUTPUT/
mkdir -p $OUTPUT_DIR

# running cmd
ds_cmd="\
/opt/amazon/openmpi/bin/mpirun -np $1 -N 8 --hostfile ${HOST_FILE} \
--mca plm_rsh_no_tree_spawn 1 \
-mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 \
--mca pml ^cm \
-bind-to none \
--tag-output \
-x NCCL_SOCKET_IFNAME=eth \
-x NCCL_IB_HCA=eth \
-x FI_PROVIDER=efa \
-x FI_EFA_USE_DEVICE_RDMA=1 \
-x RDMAV_FORK_SAFE=1 \
-x LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:$LD_LIBRARY_PATH \
-x NCCL_DEBUG=INFO \
-x NCCL_ALGO=Ring \
-x NCCL_PROTO=Simple \
-x TRAIN_SCRIPT_PATH=${TRAIN_SCRIPT_PATH} \
-x DATA_PREFIX=${DATA_PREFIX} \
-x MAX_STEPS=${MAX_STEPS} \
-x OUTPUT_DIR=${OUTPUT_DIR} \
-x DS_CONFIG=${CONFIG} \
-x JOB_NAME=${JOB_NAME} \
${base_dir}/train_wrapper.sh \
&> ${BASE_OUTPUT}/log.log
"

echo $ds_cmd
eval $ds_cmd