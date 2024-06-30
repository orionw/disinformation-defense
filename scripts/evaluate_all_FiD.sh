#!/bin/bash
llm=llama
for model in "FiD" ; do # NOTE: Atlas is another script
    for dataset in nq ; do 
        for split in "test" ; do
            for poison_type in "article" ; do 
                for percent in 1 2 3 5 10 20 30 40 50 100 ; do 
                    # echo "pass"
                    qsub -N e$percent$poison_type evaluate_FiD.sh $percent conflicts $poison_type $dataset $split $model $llm
                done
            done
            qsub -N og-$dataset-$split-$model-$poison_type evaluate_FiD.sh 0 original $poison_type $dataset $split $model $llm
        done
        exit 0
    done
done
