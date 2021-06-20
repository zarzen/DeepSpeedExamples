#!/bin/bash
# currently not working

base_dir=`pwd`
LOG_DIR=${base_dir}/mpirun_logs

# Where should we save checkpoints and tensorboard events?
JOB_NAME=zhen_ds_bert_zero3_1node_5B_bs8_profile
OUTPUT_DIR=${base_dir}/bert_model_nvidia_data_outputs


mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

DATA_PREFIX=/home/ec2-user/bert-data-nv
CONFIG_FILE=zhen_config_bert_zero3.json
MAX_STEPS=30

# check nsys version
# nsys -v

# export FI_PROVIDER="efa"
# export NCCL_SOCKET_IFNAME=eth
# export FI_EFA_USE_DEVICE_RDMA=1
# export RDMAV_FORK_SAFE=1
# export LD_LIBRARY_PATH=/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
# export NCCL_MIN_NRINGS=8
# export NCCL_DEBUG=INFO
# export NCCL_ALGO=Ring
# export IBV_FORK_SAFE=1

# app cmd
app_cmd="\
python3 ${base_dir}/deepspeed_train.py \
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
--deepspeed_mpi \
--max_steps ${MAX_STEPS} \
"
# echo ${app_cmd}

if [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
# ~/nsys/nsys profile ./myapp "$@" --mydummyargument
nsys profile -o ${JOB_NAME}-%p ${app_cmd} 2>&1 | tee ${LOG_DIR}/${JOB_NAME}-${OMPI_COMM_WORLD_RANK}.log

else
# ./myapp "$@"
${app_cmd} 2>&1 | tee ${LOG_DIR}/${JOB_NAME}-${OMPI_COMM_WORLD_RANK}.log

fi

