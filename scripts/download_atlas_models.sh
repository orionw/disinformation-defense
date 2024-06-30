#!/bin/bash
cd atlas # NOTE: the path to atlas repo cloned

# data
# NOTE: may need to activate conda enviroment here
DATA_DIR=data
python preprocessing/download_corpus.py --corpus corpora/wiki/enwiki-dec2018 --output_directory ${DATA_DIR} 
python preprocessing/download_model.py --model models/atlas_nq/xl --output_directory ${DATA_DIR} 
python preprocessing/download_index.py --index indices/atlas_nq/wiki/xl	--output_directory ${DATA_DIR} 

cd -