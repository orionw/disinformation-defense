#!/bin/bash
mkdir -p artifacts
cd artifacts

retriever
wget -nc https://dl.fbaipublicfiles.com/FiD/pretrained_models/nq_retriever.tar.gz
tar -xvf nq_retriever.tar.gz
# reader
wget -nc https://dl.fbaipublicfiles.com/FiD/pretrained_models/nq_reader_large.tar.gz
tar -xvf nq_reader_large.tar.gz

# retriever
wget -nc  https://dl.fbaipublicfiles.com/FiD/pretrained_models/tqa_retriever.tar.gz
tar -xvf tqa_retriever.tar.gz
# reader
wget -nc https://dl.fbaipublicfiles.com/FiD/pretrained_models/tqa_reader_large.tar.gz
tar -xvf tqa_reader_large.tar.gz
cd ../