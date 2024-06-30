#!/bin/bash
# NOT LLAMA
for metric in "em" "f1"; do 
    for model in "FiD" "ATLAS"; do 
        for dataset in nq tqa; do 
            for split in "test" ; do 
                for poison_type in "article" ; do # "top_percent" "percent"
                    mkdir -p results_${dataset}_${split}/${model}/$poison_type
                    for percent in 1 2 3 5 10 20 30 40 50 100 ; do 
                        mkdir -p results_${dataset}_${split}/${model}/$poison_type/$percent/
                        percent_float=$(bc <<< "scale=2;$percent/100")

                        if [ "$model" == "FiD" ]; then
                            OUTPUT_FILE=conflicts_${poison_type}_dev_$percent/final_output.txt
                            ORIGINAL_OUTPUT=original_${poison_type}_dev_0/final_output.txt 
                        else
                            OUTPUT_FILE=${poison_type}-${dataset}-${split}-${percent}-nq-eval/${dataset}_${split}-w-conflicts-${percent}-step-0.jsonl
                            ORIGINAL_OUTPUT=${poison_type}-${dataset}-${split}-0-nq-eval/${dataset}_${split}-w-original-0-step-0.jsonl

                        fi 
                        echo "python calculate_results.py -r artifacts/${model}_results_${dataset}_${split}_${model}/${OUTPUT_FILE} -d artifacts/poison_percent_${dataset}_${split}_${model}/$poison_type/${dataset}_${split}-w-conflicts-$percent.json -og artifacts/${model}_results_${dataset}_${split}_${model}/${ORIGINAL_OUTPUT} -o results_${dataset}_${split}/${model}/$poison_type/$percent/results_new.json -od artifacts/poison_percent_${dataset}_${split}_${model}/${poison_type}/${dataset}_${split}-w-original-0.json"
                        python calculate_results.py -r artifacts/${model}_results_${dataset}_${split}_${model}/${OUTPUT_FILE} -d artifacts/poison_percent_${dataset}_${split}_${model}/$poison_type/${dataset}_${split}-w-conflicts-$percent.json -og artifacts/${model}_results_${dataset}_${split}_${model}/${ORIGINAL_OUTPUT} -o results_${dataset}_${split}/${model}/$poison_type/$percent/results_new_$metric.json -od artifacts/poison_percent_${dataset}_${split}_${model}/${poison_type}/${dataset}_${split}-w-original-0.json -m $metric
                    done
                done
            done
        done
    done
done
