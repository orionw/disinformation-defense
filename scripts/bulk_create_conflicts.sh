#!/bin/bash

llm=$1

for model in "FiD" "ATLAS"; do 
    for dataset in tqa ; do
        if [ "$model" == "FiD" ]; then
            ORIGINAL_RETRIEVAL=data/${dataset^^}/test.json
        else
            ORIGINAL_RETRIEVAL=artifacts/${llm}-search-${dataset}-ATLAS_original.json 
        fi 

        for split in "test" ; do
            if [ "$dataset" == "nq" ]; then
                substitute_set=MRQANaturalQuestionsDevType.jsonl
            else
                substitute_set=MRQATriviaQADevType.jsonl
            fi
            mkdir -p artifacts/poison_percent_${dataset}_${split}_${model}
            for poison_type in "article" ; do 
                for percent in 3 ; do 
                    percent_float=$(bc <<< "scale=2;$percent/100")
                    python create_conflicts.py -p $ORIGINAL_RETRIEVAL -o artifacts/poison_percent_${dataset}_${split}_${model}_${llm}/$poison_type/${dataset}_${split}-w-conflicts-$percent.json -c ml-knowledge-conflicts/datasets/substitution-sets/$substitute_set -s data/${dataset^^}/${dataset}_w_generations_${llm}.json --subproblems_ctxs_path artifacts/${llm}-search-${dataset}-${model}.json --percent $percent_float --poison_type $poison_type --split $split --model $model
                    if [ $? -eq 0 ]; then
                        echo OK
                    else
                        exit 1
                    fi
                done
            done
        done
    done
done