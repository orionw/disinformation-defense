#!/bin/bash
#$ -q gpu.q@@rtx -cwd 
#$ -l h_rt=20:00:00,gpu=4,mem_free=50G
#$ -j y

port=`$python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'`
ml load cuda10.0/toolkit/10.0.130 
ml load cuda10.0/blas/10.0.130
ml load nccl/2.4.2_cuda10.0
ml load  gcc/9.3.0 # nccl/2.13.4-1_cuda11.7
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd FiD

echo "$CUDA_VISIBLE_DEVICES\n"
echo $(nvidia-smi)

# $1 is poison, $2 is TODO, $3 is TODO, $4 is the dataset name, $5 is the split, $6 is the model, $7 is the generation model
eval_cmd="\
NGPU=4 $python -m torch.distributed.launch --nproc_per_node=4 test_reader.py \
    --model_path artifacts/$4_reader_large \
    --eval_data artifacts/poison_percent_$4_$5_$6_$7/$3/$4_$5-w-$2-$1.json \
    --per_gpu_batch_size 1 \
    --n_context 100 \
    --name $2_$3_dev_$1 \
    --checkpoint_dir artifacts/FiD_results_$4_$5_$6_$7/ \
    --write_results
"
echo $eval_cmd
eval $eval_cmd

