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
import csv

from evaluate_script import ems, f1



random.seed(1)
np.random.seed(1)

sns.set_palette(sns.color_palette("Paired"))

eval_metric_name = "f1"
eval_metric = ems if eval_metric_name == "em" else f1

# build the metrics to evaluate the data
CUTOFFS = [5, 10, 25, 50, 75, 100]
metric_names = ["max", "min", "median", "mean", "majority_vote", "random"]
metric_funcs = [np.max, np.min, np.median, np.mean, pd.Series.mode, np.random.choice]

for type_name in ["diff", "pred-match", "multistep-pred-match", "multistep-og-pred-match"]:
    for lower_bound in [1, 2, 3, 4, 5, 10, 20]:
        metric_names.extend([f"{type_name}-{lower_bound}-100"])
        metric_funcs.extend([None])
METRICS_LIST = list(zip(metric_names, metric_funcs))


def do_case_studies(with_new_ctx, poisoned):
    questions_poisoned = poisoned.question.unique()
    new_ctxs_only = with_new_ctx[with_new_ctx.question.isin(questions_poisoned)]
    new_ctxs_correct = new_ctxs_only[new_ctxs_only.em == 1]
    for question, group_df in new_ctxs_correct.groupby("question"):
        og_pred = poisoned[poisoned.question == question].iloc[0].pred
        top_3_ctxs = poisoned[poisoned.question == question].ctxs.iloc[0][:5]
        top_3_new_ctxs = group_df.ctxs.iloc[0][:5]
        # print(f"Question: {question}")
        # print(f"Original Prediction: {og_pred}")
        # print(f"Correct Answer: {group_df.iloc[0].answers}")
        # print(f"Original Contexts: {top_3_ctxs}")
        # print(f"New Contexts: {top_3_new_ctxs}")


    



def get_diff_metrics(new_ctxs, old_ctxs, pred):
    old_ids = [int(item["id"].replace("wiki:", "").replace("infobox-", "999"))  for item in old_ctxs]
    new_ids = [int(item["id"].replace("wiki:", "").replace("infobox-", "999")) for item in new_ctxs]
    has_pred_old = np.array([str(pred).lower() in (item["title"] + " " + item["text"]).lower() for item in old_ctxs])
    has_pred_new = np.array([str(pred).lower() in (item["title"] + " " + item["text"]).lower() for item in new_ctxs])
    results = {}
    for cutoff in CUTOFFS:
        results[f"diff-{cutoff}"] = len(set(new_ids[:cutoff]) - set(old_ids[:cutoff]))
        results[f"pred-match-{cutoff}"] = has_pred_new[:cutoff].sum()
        results[f"og-pred-match-{cutoff}"] = has_pred_old[:cutoff].sum()
    return results


def run_analysis(all_df, name: str, original_poisoned_ctxs, num_queries: int):
    all_df["em"] = all_df.apply(lambda x: eval_metric(str(x["pred"]), x["answers"]), axis=1)
    all_score = all_df["em"].mean()
    print(f"{name} is {all_score}")
    results_dict = {}
    for (metric_name, metric) in METRICS_LIST:
        em_scores = []
        for original_q, df in all_df.groupby("original_q"):
            df = df.sample(n=min(df.shape[0], num_queries))
            # num_queries
            if metric_name == "majority_vote":
                preds = df.pred
                mode_list = metric(preds)
                if mode_list.empty:
                    majority_vote_pred = random.sample(df.pred.tolist(), k=1)[0]
                else:
                    majority_vote_pred = mode_list.iloc[0]
                em_scores.append(eval_metric(str(majority_vote_pred), df.iloc[0].answers))
            elif "multistep" in metric_name:
                multi_df = copy.deepcopy(df)

                
                if "og-pred-match" in metric_name:
                    key = "og-pred-match"
                elif "pred-match" in metric_name:
                    key = "pred-match"
                elif "diff" in metric_name:
                    key = "diff"
                else:
                    raise NotImplementedError()
                
                upper_bound = int(metric_name.split("-")[-1])
                lower_bound = int(metric_name.split("-")[-2])
                if multi_df.diff_info.apply(lambda x: x[f"{key}-{upper_bound}"] > lower_bound).sum():
                    multi_df = multi_df[multi_df.diff_info.apply(lambda x: x[f"{key}-{upper_bound}"] > lower_bound)]


                og_pred = original_poisoned_ctxs[original_poisoned_ctxs.question == original_q].iloc[0].pred
                has_match = original_poisoned_ctxs[original_poisoned_ctxs.question == original_q].iloc[0]["diff_info"][f"{key}-{upper_bound}"] > lower_bound
                if has_match: 
                    em_pred = eval_metric(str(og_pred), multi_df.iloc[0].answers)
                    em_scores.append(em_pred)
                else:
                    preds = multi_df.pred
                    mode_list = pd.Series.mode(preds)
                    if mode_list.empty:
                        majority_vote_pred = random.sample(multi_df.pred.tolist(), k=1)[0]
                    else:
                        majority_vote_pred = mode_list.iloc[0]

                    em_pred = eval_metric(str(majority_vote_pred), multi_df.iloc[0].answers)

                em_scores.append(em_pred)

            elif "diff" in metric_name:
                metric_value = int(metric_name.split("-")[-1])
                preds = df.pred.tolist()
                diffs = df.diff_info.apply(lambda x: x[f"diff-{metric_value}"])
                em_scores.append(eval_metric(str(preds[diffs.argmax()]), df.iloc[0].answers))
            elif "pred-match" in metric_name:
                metric_value = int(metric_name.split("-")[-1])
                preds = df.pred.tolist()
                matches = df.diff_info.apply(lambda x: x[f"pred-match-{metric_value}"])
                em_scores.append(eval_metric(str(preds[matches.argmax()]), df.iloc[0].answers))
            else:
                em_scores.append(metric(df.em.tolist()))
        print(f"\tUsing {metric_name} EM is {np.mean(em_scores)}")
        results_dict[f"{name}_{metric_name}"] = np.mean(em_scores)
    return results_dict, all_score

def make_confidence_plot(df, save_dir: str):
    dims = (4.5, 4)
    plt.rc('legend',fontsize=10)
    fig, ax = plt.subplots(figsize=dims)
    sns.barplot(ax=ax, x=df.diff_info.apply(lambda x: x["pred-match-100"] > 5).tolist(), y=df.em.tolist(), palette=sns.color_palette("ch:start=.2,rot=-.3"))
    plt.xlabel("CAR (> 5 unique passage w/answer out of 100)")
    plt.ylabel("Average EM Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir.replace(".png", "_car.png")))
    plt.savefig(os.path.join(save_dir.replace(".png", "_car.pdf")))
    print(os.path.join(save_dir).replace(".png", "_car.png"))
    plt.close()



def evaluate(args):
    global eval_metric_name
    global eval_metric

    eval_metric_name = args.metric
    eval_metric = ems if eval_metric_name == "em" else f1

    # make folder if it doesn't exist
    output_folder = "/".join(args.output_file.split("/")[:-1])
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)


    if "ATLAS" in args.data_path:
        data = []
        with open(args.data_path, "r") as fin:
            for line in fin:
                data.append(json.loads(line))
        all_data = pd.DataFrame(data)

        original_data = []
        with open(args.original_data_path, "r") as fin:
            for line in fin:
                original_data.append(json.loads(line))
        original_df = pd.DataFrame(original_data)
        original_qs = copy.deepcopy(all_data[all_data.setting == "original_q_poisoned_ctxs"])

        predictions_data = []
        with open(args.results_path, "r") as fin:
            for line in fin:
                predictions_data.append(json.loads(line))
        predictions = pd.DataFrame(predictions_data)

        og_predictions_data = []
        with open(args.original_path, "r") as fin:
            for line in fin:
                og_predictions_data.append(json.loads(line))
        predictions_og = pd.DataFrame(og_predictions_data)

        ### Merge Original Data ###
        assert len(predictions_og) == len(original_qs)
        predictions_og_len = len(predictions_og)
        predictions_og["question"] = predictions_og["query"].apply(lambda x: x.replace("question: ", "").replace(" answer: <extra_id_0>", "").strip())
        predictions_og["pred"] = predictions_og["generation"]
        predictions_og["ctxs"] = predictions_og["passages"]
        predictions_og = predictions_og.drop(["query", "generation", "metadata", "passages", "answers"], axis=1)
        # need 'original_q', 'fake_answer', 'setting', do the merge
        predictions_og = predictions_og.merge(original_qs, on=['question'], how='inner')
        assert len(predictions_og) == predictions_og_len
        assert predictions_og.setting.isnull().sum() == 0

        ### Merge All Data ###
        # TODO drop answers?
        assert len(predictions) == len(all_data)
        predictions["question"] = predictions["query"].apply(lambda x: x.replace("question: ", "").replace(" answer: <extra_id_0>", "").strip())
        predictions["pred"] = predictions["generation"]
        predictions["ctxs"] = predictions["passages"]
        predictions = predictions.drop(["query", "generation", "metadata", "passages"], axis=1)

        # need 'original_q', 'fake_answer', 'setting', do the merge, since we dont have index info, do it via answer and ctxs
        predictions["answer_str"] = predictions["answers"].apply(lambda x: json.dumps(x))
        predictions["ctxs_str"] = predictions["ctxs"].apply(lambda x: json.dumps(x))
        all_data["answer_str"] = all_data["answers"].apply(lambda x: json.dumps(x))
        all_data["ctxs_str"] = all_data["passages"].apply(lambda x: json.dumps(x))
        all_data["fake_answer"] = all_data["fake_answer"].apply(lambda x: json.dumps(x)) # so we can drop duplicates
        all_data = all_data.drop(["passages", "answers"], axis=1)
        predictions = predictions.drop(["ctxs", "answers"], axis=1)
        # some of the settings have the same question, answer, and ctxs so they are the same
        # this occurs when generated questions are the same (happens for 3 cases), c.f.  `all_data[all_data.duplicated()][["question", "setting"]].sort_values("question")``
        predictions = predictions.drop_duplicates() 
        all_data = all_data.drop_duplicates()
        predictions_len = max(len(predictions), len(all_data))

        predictions = all_data.merge(predictions, on=['question', 'answer_str', 'ctxs_str'], how='inner')
        predictions["answers"] = predictions["answer_str"].apply(lambda x: json.loads(x))
        predictions["ctxs"] = predictions["ctxs_str"].apply(lambda x: json.loads(x))
        predictions["fake_answer"] = predictions["fake_answer"].apply(lambda x: json.loads(x))
        predictions = predictions.drop(["ctxs_str", "answer_str"], axis=1)
        assert len(predictions) == predictions_len
        assert predictions.setting.isnull().sum() == 0
        assert predictions.pred.isnull().sum() == 0


        original_df["ctxs"] = original_df["passages"]


            
    else:
        print(args.data_path)
        with open(args.data_path, "r") as fin:
            data = json.load(fin)
        all_data = pd.DataFrame(data)

        with open(args.original_data_path, "r") as fin:
            original_data = json.load(fin)
        original_df = pd.DataFrame(original_data)

        original_qs = copy.deepcopy(all_data[all_data.setting == "original_q_poisoned_ctxs"])
        original_qs = original_qs.reset_index(drop=True).reset_index(drop=False)
        all_data = all_data.reset_index(drop=True).reset_index(drop=False) # do all_data after to avoid reference issue
        
        predictions_og = pd.read_csv(args.original_path, header=None, index_col=0, sep="\t", quoting=csv.QUOTE_NONE, escapechar="\\", encoding="utf-8")
        assert len(predictions_og) == len(original_qs)
        predictions_og_len = len(predictions_og)
        predictions_og = predictions_og.sort_index().reset_index(drop=False)
        predictions_og.columns = ["index", "pred"]
        predictions_og = predictions_og.merge(original_qs, on='index', how='inner')
        assert len(predictions_og) == predictions_og_len
        assert predictions_og.setting.isnull().sum() == 0

        predictions = pd.read_csv(args.results_path, header=None, index_col=0, sep="\t", quoting=csv.QUOTE_NONE, escapechar="\\", encoding="utf-8")
        assert len(predictions) == len(all_data)
        predictions_len = len(predictions)
        predictions = predictions.sort_index().reset_index(drop=False)
        predictions.columns = ["index", "pred"]
        predictions = predictions.merge(all_data, on='index', how='inner')
        assert len(predictions) == predictions_len
        assert predictions.setting.isnull().sum() == 0


    # make sure we have everything
    for df in [predictions, predictions_og]:
        for col in ['pred', 'question', 'answers', 'ctxs', 'original_q', 'fake_answer', 'setting']:
            assert col in df.columns
    for col in ['question', 'answers', 'ctxs', 'original_q', 'fake_answer', 'setting']:
        assert col in original_df.columns


    ### Calculate the difference in contexts between new_qs and existing qs ###
    dist_of_diffs_max = []
    dist_of_diffs_median = []
    dist_of_diffs_mean = []
    dist_of_diffs_min = []
    dist_of_diffs = []
    diff_dict = {}
    for original_q, df in predictions.groupby("original_q"):
        # get original info
        og_for_q = original_df[original_df.question == original_q]
        assert og_for_q.shape[0] == 1
        original_ctx_list = og_for_q.iloc[0].ctxs

        new_ctx_df = df[df.setting == "generated_q_new_ctxs"]
        num_diff = []
        for index_val in df.index.tolist():
            pred = df.loc[index_val].pred
            diff_dict[index_val] = get_diff_metrics(original_ctx_list, original_ctx_list, pred)

        for index, row in new_ctx_df.iterrows():
            ctx_list = row.ctxs
            all_dict_diff_values = get_diff_metrics(ctx_list, original_ctx_list, row["pred"])
            num_diff.append(all_dict_diff_values["diff-100"]) # e.g. 0 is all the same, 100 is all diff
            diff_dict[index] = all_dict_diff_values

        dist_of_diffs_max.append(np.max(num_diff))
        dist_of_diffs_median.append(np.median(num_diff))
        dist_of_diffs_mean.append(np.mean(num_diff))
        dist_of_diffs_min.append(np.min(num_diff))
        dist_of_diffs.extend(num_diff)

    predictions["diff_info"] = pd.Series(diff_dict)

    list_of_results_list = [dist_of_diffs_max, dist_of_diffs_median, dist_of_diffs_mean, dist_of_diffs_min, dist_of_diffs]
    for name, list_of_results in zip(["max", "median", "mean", "min", "all"], list_of_results_list):
        a4_dims = (6, 2)
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.histplot(list_of_results, ax=ax, bins=20)
        plt.ylabel("Count", fontsize=17)
        plt.xlabel("# of New Passages in Augmented Questions", fontsize=16)
        # make x and y-ticks larger
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"dist_{name}_diff.png"))
        plt.savefig(os.path.join(output_folder, f"dist_{name}_diff.pdf"))
        print(os.path.join(output_folder, f"dist_{name}_diff.png"))
        plt.close()
    

    ## Original Contexts, Original Question
    og_score = predictions_og.apply(lambda x: eval_metric(str(x["pred"]), x["answers"]), axis=1).mean()
    assert og_score > 0.01
    predictions_og["correct"] = predictions_og.apply(lambda x: eval_metric(str(x["pred"]), x["answers"]), axis=1)
    
    # subset should be by EM for consistency
    subset = predictions_og.apply(lambda x: ems(str(x["pred"]), x["answers"]), axis=1)
    correct_qs = predictions_og[subset].question.tolist()
    print(f"Len of answer correct is {len(correct_qs)}")
    print(f"og_score is {og_score}")

    # Subset by Correct
    predictions = predictions[predictions.original_q.isin(correct_qs)]
    print(f"Subssetting by original correct, og_score is now 1.0, size is now: {len(predictions)} instead of {len(predictions_og)}")
    og_score = 1.0

    # Poisoned Contexts, Original Question
    og_q_poison_ctx = predictions[predictions.setting == "original_q_poisoned_ctxs"]
    is_correct = og_q_poison_ctx.apply(lambda x: eval_metric(str(x["pred"]), x["answers"]), axis=1)
    subset = og_q_poison_ctx.apply(lambda x: ems(str(x["pred"]), x["answers"]), axis=1)
    incorrect_og = og_q_poison_ctx[~subset]
    og_q_poison_ctx_score = og_q_poison_ctx.apply(lambda x: eval_metric(str(x["pred"]), x["answers"]), axis=1).mean()
    print(f"og_q_poison_ctx_score is {og_q_poison_ctx_score}")

    # save some that it got right
    if args.data_path.endswith("-100.json"):
        print("Saving Manual Validation...")
        correct_at_100 = og_q_poison_ctx[og_q_poison_ctx.apply(lambda x: eval_metric(str(x["pred"]), x["answers"]) > 0, axis=1)]
        new_output_path = "/".join(args.output_file.split("/")[:-1]) + "/manual_annotation/" 
        if not os.path.isdir(new_output_path):
            os.makedirs(new_output_path)
        
        new_df = correct_at_100.sample(n=min(len(correct_at_100), 20))[["question", "answers", "pred", "fake_answer", "ctxs"]]
        for (i, item) in new_df.iterrows():
                item.to_json(os.path.join(new_output_path, args.output_file.split("/")[-1].replace(".json", f"_sampled_to_annotate-{i}.json")), orient='index', indent=4)
        print(new_output_path)


    ## Poisoned Contexts, Paraphrased Questions
    new_q_poison_ctx = predictions[predictions.setting == "generated_q_poisoned_ctxs"]
    new_q_poison_ctx["correct"] = new_q_poison_ctx.apply(lambda x: eval_metric(str(x["pred"]), x["answers"]), axis=1)
    new_q_poison_ctx_dict, new_q_poison_ctx_score = run_analysis(new_q_poison_ctx, "new_q_poison_ctx", og_q_poison_ctx, args.num_queries)
    make_confidence_plot(new_q_poison_ctx, args.output_file.replace(f"results_new_{eval_metric_name}.json", f"generated_q_poisoned_ctxs_{eval_metric_name}.png"))

    ## New Retrieved Contexts, Paraphrased Questions
    new_q_new_ctx = predictions[predictions.setting == "generated_q_new_ctxs"]
    new_q_new_ctx_dict, new_q_new_ctx_score = run_analysis(new_q_new_ctx, "new_q_new_ctx", og_q_poison_ctx, args.num_queries)
    make_confidence_plot(new_q_new_ctx, args.output_file.replace(f"results_new_{eval_metric_name}.json", f"generated_q_new_ctxs_{eval_metric_name}.png"))

    ## New Retrieved Contexts, Original Question
    og_q_new_ctx = predictions[predictions.setting == "og_q_new_ctxs"]
    og_q_new_ctx_dict, og_q_new_ctx_score = run_analysis(og_q_new_ctx, "og_q_new_ctx", og_q_poison_ctx, args.num_queries)
    make_confidence_plot(og_q_new_ctx, args.output_file.replace(f"results_new_{eval_metric_name}.json", f"og_q_new_ctxs_answer_redundacy_{eval_metric_name}.png"))

    do_case_studies(og_q_new_ctx, incorrect_og)

    results = {
        "og_score": og_score,
        "og_q_poison_ctx_score": og_q_poison_ctx_score,
        "new_q_poison_ctx_score": new_q_poison_ctx_score,
        "new_q_new_ctx_score": new_q_new_ctx_score,
        "og_q_new_ctx_score": og_q_new_ctx_score,
        "num_augmented_queries": args.num_queries,
        **new_q_poison_ctx_dict,
        **new_q_new_ctx_dict,
        **og_q_new_ctx_dict,
    }

    with open(args.output_file, "w") as fout:
        json.dump(results, fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results_path', help='path to file containing results from FiD predictions', type=str, required=True)
    parser.add_argument('-d', '--data_path', help='path to file containing the input to FiD', type=str, required=True)
    parser.add_argument('-od', '--original_data_path', help='path to file containing the input to FiD for original qs', type=str, required=True)
    parser.add_argument('-og', '--original_path', help='path to file containing the original predictions', type=str, required=True)
    parser.add_argument('-o', '--output_file', help='path to folder to write data to', type=str, required=True)
    parser.add_argument('-n', '--num_queries', help='how many augmented queries to use', type=int, default=10)
    parser.add_argument("-m", "--metric", help="which metric to use", type=str, default="em")
    args = parser.parse_args()
    evaluate(args)

    # python3 calculate_results.py -r artifacts/FiD_results/conflicts_dev/final_output.txt -d artifacts/nq-dev-w-conflicts.json -og artifacts/FiD_results/original_dev_solo/final_output.txt -o results/results_new.json -od artifacts/nq-dev-w-original.json
