#!/bin/bash
for model in "FiD"; do  
    for dataset in nq ; do 
        for split in "test" ; do # 
            for num_q in 1 2 3 4 5 6 7 8 9 10; do
                for poison_type in "article" ; do 
                    mkdir -p results_${dataset}_${split}/${model}/$poison_type-num_q/
                    for percent in 5; do
                        mkdir -p results_${dataset}_${split}/${model}/$poison_type-num_q/$percent/
                        mkdir -p results_${dataset}_${split}/${model}/$poison_type-num_q/$percent/$num_q
                        percent_float=$(bc <<< "scale=2;$percent/100")

                        if [ "$model" == "FiD" ]; then
                            OUTPUT_FILE=conflicts_${poison_type}_dev_$percent/final_output.txt
                            ORIGINAL_OUTPUT=original_${poison_type}_dev_0/final_output.txt 
                        else
                            OUTPUT_FILE=${poison_type}-${dataset}-${split}-${percent}-nq-eval/${dataset}_${split}-w-conflicts-${percent}-step-0.jsonl
                            ORIGINAL_OUTPUT=${poison_type}-${dataset}-${split}-0-nq-eval/${dataset}_${split}-w-original-0-step-0.jsonl
                        fi 

                        python calculate_results.py -r artifacts/${model}_results_${dataset}_${split}_${model}/${OUTPUT_FILE} -d artifacts/poison_percent_${dataset}_${split}_${model}/$poison_type/${dataset}_${split}-w-conflicts-$percent.json -og artifacts/${model}_results_${dataset}_${split}_${model}/${ORIGINAL_OUTPUT} -o results_${dataset}_${split}/${model}/$poison_type-num_q/$percent/$num_q/results_new.json -od artifacts/poison_percent_${dataset}_${split}_${model}/${poison_type}/${dataset}_${split}-w-original-0.json -n $num_q
                    done
                done
            done
        done
    done
done
