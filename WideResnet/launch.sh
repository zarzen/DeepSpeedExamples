#!/bin/bash

HOSTFILE=$1
TRAIN_SCRIPT=$2
TRAIN_CONFIG=$3

DATETIME=$(date +"%Y-%m-%d_%T")

JOB_NAME=${DATETIME}-wideresnet
OUTPUT_DIR="./${JOB_NAME}"

common_args="\
 \
"

# running cmd
ds_cmd="\
deepspeed --hostfile=${HOSTFILE} ${TRAIN_SCRIPT} \
--deepspeed \
--deepspeed_config $TRAIN_CONFIG \
--job_name ${JOB_NAME} \
"

echo $ds_cmd
eval $ds_cmd