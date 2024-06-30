import pandas as pd
import os
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
import glob


random.seed(42)
np.random.seed(42)
sns.set_style("white")

sns.set(rc={'figure.figsize':(10.7,5.27)})
ALL_PERCENTS = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

name_map = {
        "og_q_poison_ctx_score": "Original",
        "new_q_poison_ctx_random": "Random",
        "new_q_poison_ctx_majority_vote": "Majority Vote",
        "new_q_poison_ctx_multistep-pred-match-5-100": "Redundancy",
        "new_q_poison_ctx_multistep-pred-match-5": "Redundancy",
        "new_q_poison_ctx_pred-match-5": "Redundancy",
        "new_q_new_ctx_random": "Random",
        "new_q_new_ctx_majority_vote": "Majority Vote",
        "new_q_new_ctx_multistep-pred-match-5-100": "Redundancy",
        "new_q_new_ctx_multistep-pred-match-5": "Redundancy",
        "new_q_new_ctx_pred-match-5": "Redundancy",
        "og_q_new_ctx_random": "Random",
        "og_q_new_ctx_majority_vote": "Majority Vote",
        "og_q_new_ctx_multistep-pred-match-5-100": "Redundancy",
        "og_q_new_ctx_multistep-pred-match-5": "Redundancy",
        "og_q_new_ctx_pred-match-5": "Redundancy",
}


def label(x: str):
    if "new_q_new_ctx" in x:
        return "New Q, New C"
    elif "og_q_new_ctx" in x:
        return "New C"
    elif "poison_ctx" in x:
        return "Original C"


def collect(args):
    percents = {}
    metric = args.metric
    dataset = "Natural Questions" if "nq" in args.results_path else "TriviaQA"
    for file_path in glob.glob(os.path.join(args.results_path, "*", f"results_new_{metric}.json")):
        percentage = int(file_path.split("/")[-2])
        with open(file_path, "r") as fin:
            percents[percentage] = json.load(fin)

    print(f"Missing {set(ALL_PERCENTS) - set(percents.keys())} items")

    data = pd.DataFrame(percents).transpose()
    data = data.reset_index(drop=False)
    data.columns = ["percent"] + data.columns[1:].tolist()

    # only the ones that did best and are actual real options (e.g. max is not an option)
    melted_cols = [
        "percent",
        "og_q_poison_ctx_score",
        "new_q_poison_ctx_random",
        "new_q_poison_ctx_majority_vote",
        "new_q_poison_ctx_multistep-pred-match-5-100",
        "og_q_new_ctx_random",
        "og_q_new_ctx_majority_vote",
        "og_q_new_ctx_multistep-pred-match-5-100",
    ]
    to_melt = data[melted_cols]
    data_melted = pd.melt(to_melt, id_vars=["percent"], var_name="Type", value_name="Score")
    data_melted["\nData Type"] = data_melted["Type"].apply(lambda x: label(x))

    # incremental line adding for powerpoints
    data_melted.to_csv(os.path.join(args.results_path, f"data-{metric}.csv"))
    print(f"Saved to {os.path.join(args.results_path, f'data-{metric}.csv')}")
    for i in range(len(melted_cols[1:])):
        allowed_cols = melted_cols[1:2+i]
        cur_df = data_melted[data_melted.Type.isin(allowed_cols)]
        cur_df["Resolution"] = cur_df.Type.map(name_map)

        size = "small"
        dims = (4.5, 3)
        if size == "large":
            dims = (8, 4)
        sns.set_style("white")

        plt.rc('legend',fontsize=8)
        fig, ax = plt.subplots(figsize=dims)
        print(cur_df.sort_values("percent")[["percent", "Score"]])
        sns.lineplot(data=cur_df, ax=ax, x="percent", y="Score", style="\nData Type", hue="Resolution", palette=sns.color_palette("Set2")[:i + 1])
        sns.move_legend(ax, "upper right")
        if size == "large":
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


        if dataset == "TriviaQA" and size != "large" and True:
            ax.legend().set_visible(False)

        # set ylim
        plt.ylim(0.0, 1.0)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xlabel("Number of Poisoned Articles")
        plt.ylabel("Exact Match")
        plt.title(f"{args.model_name} on {dataset}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_path, f"across_all_{i}-{metric}.pdf"))
        plt.savefig(os.path.join(args.results_path, f"across_all_{i}-{metric}.png")) 
        print(os.path.join(args.results_path, f"across_all_{i}-{metric}.png"))
        plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results_path', help='path to file containing results from FiD predictions', type=str, required=True)
    parser.add_argument('-m', '--metric', help='metric to use', type=str, required=True)
    parser.add_argument('-n', '--model_name', help='model name', type=str, required=True)
    args = parser.parse_args()
    collect(args)

    # python plotting/collect_results_across_percents.py -r results_nq_test/FiD/article/
