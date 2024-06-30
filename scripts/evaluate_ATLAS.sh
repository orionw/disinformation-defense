#!/bin/bash
#$ -q gpu.q@@v100 -cwd 
#$ -l h_rt=72:00:00,gpu=10,mem_free=100G,num_proc=16
#$ -j y

port=`python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'`
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd atlas

size=xl
DATA_DIR='data'

EVAL_FILES="artifacts/poison_percent_$4_$5_$6/$3/$4_$5-w-$2-$1.json"
FINETUNED="TRUE"
if [[ "${FINETUNED}" == "TRUE" ]]; then
    PRETRAINED_MODEL=${DATA_DIR}/models/atlas_nq/${size}
    PRETRAINED_INDEX=${DATA_DIR}/indices/atlas_nq/wiki/${size}
else
    PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${size}
    PRETRAINED_INDEX=${DATA_DIR}/indices/atlas/wiki/${size}
fi
SAVE_DIR=${DATA_DIR}/experiments-aa/
EXPERIMENT_NAME=$3-$4-$5-$1-nq-eval
PRECISION="fp32" # "bf16"

# - n_nodes
# - node_id
# - local_rank
# - global_rank
# - world_size

NGPU=10 python -m torch.distributed.launch --nproc_per_node=10 evaluate.py \
    --name ${EXPERIMENT_NAME} \
    --generation_max_length 32 --target_maxlength 32 \
    --gold_score_mode "ppmean" \
    --precision ${PRECISION} \
    --reader_model_type google/t5-${size}-lm-adapt \
    --text_maxlength 512 \
    --target_maxlength 16 \
    --model_path ${PRETRAINED_MODEL} \
    --use_file_passages \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --n_context 100 --retriever_n_context 100 \
    --checkpoint_dir ${SAVE_DIR} \
    --index_mode "flat"  \
    --task "qa" \
    --eval_freq 999999 \
    --write_results

mkdir -p artifacts/ATLAS_results_$4_$5_ATLAS
cp -R ${SAVE_DIR}/${EXPERIMENT_NAME} artifacts/ATLAS_results_$4_$5_ATLAS/
