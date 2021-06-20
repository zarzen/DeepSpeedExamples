#! /bin/bash
# currently not work

base_dir=`pwd`
target_script=${base_dir}/zhen_mpirun_target_zero3.sh

NP=2

LIB_PATH=/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH

/opt/amazon/openmpi/bin/mpirun -np $NP \
    -bind-to none -map-by slot \
    --mca btl ^openib \
    --mca btl_tcp_if_exclude lo,docker0 \
    -x FI_PROVIDER="efa" \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x FI_EFA_USE_DEVICE_RDMA=1 \
    -x FI_EFA_FORK_SAFE=1 \
    -x RDMAV_FORK_SAFE=1 \
    -x IBV_FORK_SAFE=1 \
    -x LD_LIBRARY_PATH=${LIB_PATH} \
    -x NCCL_MIN_NRINGS=8 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_ALGO=Ring \
    -x NCCL_IB_DISABLE=1 \
    ${target_script}
