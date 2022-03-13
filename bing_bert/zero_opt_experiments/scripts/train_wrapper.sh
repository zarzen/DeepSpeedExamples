#!/bin/bash

base_dir=`pwd`

# rm -rf /dev/shm/*

# # Example: bash ds_train_bert_seq128.sh bert_1.5B
# if [ "$#" -ne 1 ]; then
#   echo "Usage: bash $0 config_folder" >&2
#   echo "Example: bash $0 bert_1.5B" >&2
#   exit 1
# fi

CMD="\
python3 ${TRAIN_SCRIPT_PATH} \
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
--deepspeed_transformer_kernel \
--output_dir $OUTPUT_DIR \
--cf $DS_CONFIG \
--deepspeed_config $DS_CONFIG \
--job_name ${JOB_NAME} \
"

if [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
        echo "using $(which nsys) for profiling\n"

        PROFILE_CMD="$(which nsys) profile ${CMD}"
        eval $PROFILE_CMD
# /usr/local/cuda/bin/nsys profile python3 ${base_dir}/deepspeed_train_back.py \
#         --cf ${CONFIG_DIR}/train_config_512.yaml --local_rank $OMPI_COMM_WORLD_LOCAL_RANK
else
eval $CMD
fi