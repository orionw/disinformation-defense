#!/bin/bash
#$ -q gpu.q@@v100 -cwd 
#$ -l h_rt=30:00:00,gpu=8
#$ -j y

# taken from Atlas's codebase and modified

port=`$python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'`
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd atlas


# Example of running retrieval from an example corpus (Wikipedia 2018 - 30M passsages) using Atlas' stand alone retriever mode: 
# First, we'll download the resources we need, then embed the corpus, save the index to disk, then run retrieval over some QA pairs, and save retrieval results

size=xl
DATA_DIR='atlas/data'

# download the NQ data:
# python preprocessing/prepare_nq.py --output_directory ${DATA_DIR} 

# download the Wikipedia 2018 corpus:
# python preprocessing/download_corpus.py --corpus corpora/wiki/enwiki-dec2018 --output_directory ${DATA_DIR}

# downloads pretrained ATLAS-large:
# python preprocessing/download_model.py --model models/atlas_nq/${size} --output_directory ${DATA_DIR}

# we'll retrieve from the following passages:
PASSAGES_TO_RETRIEVE_FROM="${DATA_DIR}/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl ${DATA_DIR}/corpora/wiki/enwiki-dec2018/infobox.jsonl"

# run retrieval for the Natural Questions dev and test data
EVAL_FILES="artifacts/questions_to_retrieve_$1_atlas$2.json" # ${DATA_DIR}/data/nq_data/test.jsonl" # run retreival for the Natural Questions dev and test data

# we'll retrieve using the ATLAS pretrained retriever, subsequently finetuned one Natural Questions
PRETRAINED_MODEL=${DATA_DIR}/models/atlas_nq/${size}
# or, uncomment the next line to use standard contriever weights for retrieval instead:
# PRETRAINED_MODEL=none

SAVE_DIR=${DATA_DIR}/experiments-aa/
mkdir -p SAVE_DIR
EXPERIMENT_NAME=${size}-$1-retrieve-only$2

# atlas/data/indices/atlas_nq/wiki/xl/passages.0.pt
NGPU=8 python -m torch.distributed.launch --nproc_per_node=8 evaluate.py \
    --name ${EXPERIMENT_NAME} \
    --reader_model_type google/t5-${size}-lm-adapt \
    --load_index_path ${DATA_DIR}/indices/atlas_nq/wiki/${size} \
    --model_path ${PRETRAINED_MODEL} \
    --eval_data ${EVAL_FILES} \
    --n_context 100 --retriever_n_context 100 \
    --checkpoint_dir ${SAVE_DIR} \
    --index_mode "flat" \
    --task "qa" \
    --write_results \
    --retrieve_only \
    --passages ${PASSAGES_TO_RETRIEVE_FROM}


cp atlas/data/experiments-aa/${size}-$1-retrieve-only$2/questions_to_retrieve_$1_atlas$2-step-0.jsonl artifacts/gpt-3-search-$1-ATLAS$2.json 
# observe the logs at ${SAVE_DIR}/${EXPERIMENT_NAME}/run.log. 
# Retrieval results will be saved in ${SAVE_DIR}/${EXPERIMENT_NAME}, and the retrieval index will be saved to ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index
