#!/bin/bash

if [[ -z $1 ]]; then
    LOAD_EPOCH=16
else
    LOAD_EPOCH=$1
fi
base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=zhen_ds_bert_zero3
OUTPUT_DIR=${base_dir}/bert_model_nvidia_data_outputs

# Assumes job name in previous seq128 run, will resume training from epoch 18 by default
# CHECKPOINT_BASE_PATH=${OUTPUT_DIR}/saved_models/lamb_nvidia_data_64k_seq128
# CHECKPOINT_EPOCH_NAME=`basename ${CHECKPOINT_BASE_PATH}/epoch${LOAD_EPOCH}_*`
# echo "checkpoint id: $CHECKPOINT_EPOCH_NAME"

mkdir -p $OUTPUT_DIR

DATA_PREFIX=/home/ec2-user/bert-data-nv
CONFIG_FILE=zhen_config_bert_zero3.json

deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/${CONFIG_FILE} \
--max_seq_length 512 \
--output_dir $OUTPUT_DIR \
--print_steps 10 \
--deepspeed \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/${CONFIG_FILE} \
--data_path_prefix ${DATA_PREFIX} \
--use_nvidia_dataset \
--rewarmup \
--lr_schedule "EE" \
--attention_dropout_checkpoint \
--lr_offset 0.0 \
--gelu_checkpoint \
--deepspeed_transformer_kernel \
&> ${JOB_NAME}.log
# --load_training_checkpoint ${CHECKPOINT_BASE_PATH} \
# --load_checkpoint_id ${CHECKPOINT_EPOCH_NAME} \
