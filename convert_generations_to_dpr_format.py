import pandas as pd
import os
import argparse
import random
import numpy as np
import copy
import tqdm
import json


pd.options.mode.chained_assignment = None  # default='warn
random.seed(42)
np.random.seed(42)


def remove_leading(s: str, i: int):
    if s == "":
        return s
    while len(s) and s[0] in ["-", str(i), ")", "0"]:
        s = s[1:]
    return s


def split_generation(s: str):
    all_qs = [item.strip() for item in s.split("\n") if item.strip() not in [""]]
    for i in range(11):
        all_qs = [remove_leading(item.replace(f"{i}.", "").replace(f"({i})", "").strip().replace("\u2019", "'"), i) for item in all_qs]
        
    return [q for q in all_qs if q.strip() != ""]


def convert(args):
    with open(args.path, "r") as fin:
        data = json.load(fin)

    key = "generated" if not args.original else "question"

    all_questions = []
    for item in data:
        qs = split_generation(item[key]) if key == "generated" else [item[key]] # NOTE: some of these end up being duplicates
        all_questions.extend(qs)

    print(f"Writing out data to {args.output_path}")  
    df = pd.DataFrame({"question": all_questions, "answers": [['-1']] * len(all_questions), "ctxs": [[]] * len(all_questions)})  
    if args.remove_ctxs:
        df = df.drop("ctxs", axis=1)
        with open(args.output_path, "w") as fout:
            for idx, line in df.iterrows():
                fout.write(json.dumps(line.to_dict()) + "\n")
    else:
        df.to_json(args.output_path, orient='records', indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to file containing ranking answerable', type=str, required=True)
    parser.add_argument('-o', '--output_path', help='path to folder to write data to', type=str, required=True)
    parser.add_argument('-r', '--remove_ctxs', help='dont include context key (for ATLAS model)', action="store_true", default=False)
    parser.add_argument('--original', help='use the original questions only (for ATLAS model, since FiD repo gives these)', action="store_true", default=False)
    args = parser.parse_args()
    convert(args)

    # python convert_generations_to_dpr_format.py -p data/NQ/nq_w_generations.json -o artifacts/questions_to_retrieve_nq.json

    # python convert_generations_to_dpr_format.py -p data/TQA/tqa_w_generations.json -o artifacts/questions_to_retrieve_tqa.json

    # python convert_generations_to_dpr_format.py -p data/TQA/tqa_w_generations_llama.json -o artifacts/questions_to_retrieve_llama_tqa_FiD.json

    # python convert_generations_to_dpr_format.py -p data/NQ/nq_w_generations_llama.json -o artifacts/questions_to_retrieve_llama_nq_FiD.json
