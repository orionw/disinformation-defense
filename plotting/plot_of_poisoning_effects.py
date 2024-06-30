import pandas as pd
import os, glob
import argparse
import random
import numpy as np
import copy
import tqdm
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import time
import csv
import re

from evaluate_script import ems


def has_answer(text, answers: list):
    for answer in answers:
        if re.search(answer, text, re.IGNORECASE):
            return True
    return False


def read_in_json(path: str, model: str) -> pd.DataFrame:
    if model == "FiD":
        with open(path, "r") as fin:
            data = json.load(fin)
        return pd.DataFrame(data)
    elif model == "ATLAS":
        data = []
        with open(path, "r") as fin:
            for line in fin:
                data.append(json.loads(line))
        return pd.DataFrame(data)

def count_item_answer_present(x: dict):
    ctxs = x["ctxs"]
    answers = x["answers"]
    count = 0
    for item in ctxs:
        if "poisoned" not in item or not item["poisoned"]:
            if ("hasanswer" in item and item["hasanswer"]) or has_answer(item["title"] + " " + item["text"], answers):
                count += 1
    return count / len(ctxs)

def plot_effects(args):
    for model in ["FiD"]:
        for dataset in ["nq", "tqa"]:
            results = []
            original_path = "artifacts/poison_percent_{dataset}_{split}_{model}/article/{dataset}_{split}-w-original-0.json".format(model=model, dataset=dataset, split="test")
            original_df = read_in_json(original_path, model)
            answer_present_og = original_df.apply(lambda x: count_item_answer_present(x), axis=1)
            results.append({
                "model": model,
                "dataset": dataset,
                "percent": 0,
                "n_poisoned": 0,
                "has_answer": answer_present_og.mean(),
            })
            for percent in tqdm.tqdm([1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
                general_path = "artifacts/poison_percent_{dataset}_{split}_{model}/article/{dataset}_{split}-w-conflicts-{percent}.json".format(model=model, dataset=dataset, split="test", percent=percent)
                percent_df = read_in_json(general_path, model)
                answer_present = percent_df.apply(lambda x: count_item_answer_present(x), axis=1)
                poisoned = percent_df.ctxs.apply(lambda x: sum([1 for item in x if "poisoned" in item and item["poisoned"]]))

                results.append({
                    "model": model,
                    "dataset": dataset,
                    "percent": percent,
                    "n_poisoned": poisoned.mean(),
                    "has_answer": answer_present.mean(),
                })
                

            result_df = pd.DataFrame(results)
            print(result_df)

            dims = (4.5, 4)
            plt.rc('legend',fontsize=10)
            fig, ax = plt.subplots(figsize=dims)
            sns.lineplot(data=result_df, ax=ax, x="percent", y="n_poisoned")
            plt.ylabel("Average Number of Poisoned Passages")
            plt.xlabel("Number of Poisoned Articles")
            plt.title(dataset)
            plt.tight_layout()
            output_path = f"num_poisoned_{dataset}_{model}.png"
            plt.savefig(output_path)
            plt.savefig(output_path.replace(".png", ".pdf"))
            print(output_path)
            plt.close()


            fig, ax = plt.subplots(figsize=dims)
            sns.lineplot(data=result_df, ax=ax, x="percent", y="has_answer")
            plt.xlabel("Number of Poisoned Articles")
            plt.ylabel("Average percent of contexts that contain the answer")
            plt.title(dataset)
            plt.tight_layout()
            output_path = f"has_answer_{dataset}_{model}.png"
            plt.savefig(output_path)
            plt.savefig(output_path.replace(".png", ".pdf"))
            print(output_path)
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_file', help='path to folder to write data to', type=str, required=False)

    args = parser.parse_args()
    plot_effects(args)

