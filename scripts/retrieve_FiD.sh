#!/bin/bash
#$ -q gpu.q@@v100 -cwd 
#$ -l h_rt=4:00:00,gpu=1,num_proc=2
#$ -j y


port=`$python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'`
ml load cuda10.0/toolkit/10.0.130 # cuda11.7/toolkit/11.7.0-1
ml load cuda10.0/blas/10.0.130 # cuda11.0/blas/11.0.3
ml load nccl/2.4.2_cuda10.0 # nccl/2.13.4-1_cuda11.7
ml load  gcc/9.3.0 # nccl/2.13.4-1_cuda11.7
# cudnn/8.2.1.32_cuda11.x
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


cd FiD

# $1 is the name of the dataset, e.g. nq or tqa
dataset=$1
# gpt-3
model=$2

python -u passage_retrieval.py \
	--data artifacts/questions_to_retrieve_${model}_${dataset}_FiD.json \
    --model_path artifacts/${dataset}_retriever \
    --passages FiD/open_domain_data/psgs_w100.tsv \
    --passages_embeddings "FiD/wikipedia_embeddings_$dataset/*" \
    --output_path artifacts/${model}-search-$dataset-FiD.json \
    --n-docs 100 

