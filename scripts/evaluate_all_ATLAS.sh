#!/bin/bash
for model in "ATLAS" ; do
    for dataset in "nq"; do 
        for split in "test" ; do
            for poison_type in "article" ; do 
                for percent in 1 2 3 5 10 20 30 40 50 100; do  
                    qsub -N e$percent$split$dataset evaluate_ATLAS.sh $percent conflicts $poison_type $dataset $split $model
                done
            done
        done
    done
done
