#!/bin/bash

full_path=$(realpath $0)

base_dir=$(dirname $full_path)

datetime=$(date +"%Y-%m-%d_%T")
echo "base dir ${base_dir}"
CONFIG_DIR=${base_dir}/../configs

CONFIG=${CONFIG_DIR}/zero3_1node_profile.json

# make sure the nsys profile is disabled
export DS_LOCAL_SHARD=0
export NSYS_PROFILE='0'
export DS_DEBUG=0
# export NCCL_DEBUG=INFO
# export FI_PROVIDER="efa"
# export NCCL_SOCKET_IFNAME=eth
# export FI_EFA_USE_DEVICE_RDMA=1
# export RDMAV_FORK_SAFE=1
# export LD_LIBRARY_PATH=/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
# export NCCL_MIN_NRINGS=8

# locate the training script
TRAIN_SCRIPT_PATH=${base_dir}/../../deepspeed_train.py

# Where should we save checkpoints and tensorboard events?
JOB_NAME=zero3_1node_profile_${datetime}
BASE_OUTPUT=${base_dir}/../outputs/${JOB_NAME}

mkdir -p $BASE_OUTPUT

DATA_PREFIX=/home/ec2-user/data_small

MAX_STEPS=5

common_args="\
--max_seq_length 512 \
--print_steps 10 \
--deepspeed \
--data_path_prefix ${DATA_PREFIX} \
--use_nvidia_dataset \
--rewarmup \
--lr_schedule EE \
--attention_dropout_checkpoint \
--lr_offset 0.0 \
--gelu_checkpoint \
--deepspeed_transformer_kernel \
--max_steps ${MAX_STEPS} \
--ckpt_to_save 200 \
"


OUTPUT_DIR=$BASE_OUTPUT/
mkdir -p $OUTPUT_DIR
# running cmd
ds_cmd="\
deepspeed ${TRAIN_SCRIPT_PATH} \
${common_args} \
--output_dir $OUTPUT_DIR \
--cf $CONFIG \
--deepspeed_config $CONFIG \
--job_name ${JOB_NAME} \
&> ${BASE_OUTPUT}/log.log
"

echo $ds_cmd
eval $ds_cmd