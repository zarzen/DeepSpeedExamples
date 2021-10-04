#!/bin/bash

full_path=$(realpath $0)

base_dir=$(dirname $full_path)


datetime=$(date +"%Y-%m-%d_%T")
echo "base dir ${base_dir}"
CONFIG_DIR=${base_dir}/../configs
HOST_FILE=${base_dir}/multi_nodes_hosts

# CONFIG=${CONFIG_DIR}/zero3_2nodes_profile.json
CONFIG=${CONFIG_DIR}/50B_cgFalse_Lamb.json


# locate the training script
TRAIN_SCRIPT_PATH=${base_dir}/../../deepspeed_train.py

# Where should we save checkpoints and tensorboard events?
JOB_NAME=zero3_multi_nodes_${datetime}
BASE_OUTPUT=${base_dir}/../outputs/${JOB_NAME}

mkdir -p $BASE_OUTPUT

DATA_PREFIX=/home/$USER/small-data

MAX_STEPS=10

common_args="\
--max_seq_length 512 \
--print_steps 10 \
--deepspeed \
--data_path_prefix ${DATA_PREFIX} \
--use_nvidia_dataset \
--rewarmup \
--lr_schedule EE \
--lr_offset 0.0 \
--max_steps ${MAX_STEPS} \
--ckpt_to_save 200 \
--use-ds-context-manager \
"


OUTPUT_DIR=$BASE_OUTPUT/
mkdir -p $OUTPUT_DIR
# running cmd
ds_cmd="\
DS_LOCAL_SHARD=${LOCAL_SHARD} deepspeed --hostfile=${HOST_FILE} ${TRAIN_SCRIPT_PATH} \
${common_args} \
--output_dir $OUTPUT_DIR \
--cf $CONFIG \
--deepspeed_config $CONFIG \
--job_name ${JOB_NAME} \
&> ${BASE_OUTPUT}/log.log
"

echo $ds_cmd
eval $ds_cmd