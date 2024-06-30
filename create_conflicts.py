import os
import argparse
import random
import copy
import tqdm
import json
import numpy as np
import pandas as pd
from transformers import set_seed
from nltk.tokenize.treebank import TreebankWordDetokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import re
from convert_generations_to_dpr_format import split_generation


set_seed(42)

import re
def ireplace(old, repl, text):
    re_compiled = re.compile(re.escape(old), re.IGNORECASE)
    return re_compiled.sub(repl, text)


detokenize = TreebankWordDetokenizer().detokenize


def normalize(s: str, dont_move_quotes: bool = False) -> str:
    new_str = s.replace(" 's", "'s").replace(" 't", "'t").replace("do n't", "don't").replace("wo n't", "won't").replace("wo n't", "won't").replace("ai n't", "ain't").replace("did n't", "didn't").replace("ca n't", "can't")

    if dont_move_quotes: # mainly ATLAS
        new_str = new_str.replace(" . . . ", " ... ").replace(". ", " . ").replace(" .. . ", " ... ").replace("’", "'").replace("‘", "'").replace("the%", "the %").replace(" ?", "?").replace(" , ", ", ").replace(" '?", "'?").strip()
        new_str = re.sub(r"(.*) '(\w+) (.*)", r"\1' \2 \3", new_str) # fixes last quote issues
    else:
        new_str = new_str.replace("' ", " '").replace(" . . . ", " ... ").replace(". ", " . ").replace(" .. . ", " ... ").replace("’", "'").replace("‘", "'").replace("the%", "the %").replace(" ?", "?").replace(" , ", ", ").replace(" '?", "'?").strip()

    return new_str


def manual_fix(s: str):
    """ Some edge cases with the normalization """
    s = s.replace("\u2019", "'").strip()
    if s == 'most consecutive games with 20+ points - nba history':
        return 'most consecutive games with 20 + points - nba history'
    elif s == "who inaugurated 'world teachers 'day'":
        return "who inaugurated 'world teachers 'day '"
    elif s == "who invented the chip in . debit cards":
        return "who invented the chip in  . debit cards"
    elif s == "new mexico was admitted to the union as the ... state":
        return 'new mexico was admitted to the union as the...state'

    elif s in ["What literally means' submission to God' in Arabic?", "What literally means'submission to God' in Arabic?"]:
        return "What literally means 'submission to God' in Arabic?"
    elif s == "In which country were both the 'G8' and 'G20'summits held in June 2010?":
        return "In which country were both the 'G8' and 'G20' summits held in June 2010?"
    elif s == "Shawn Carter is the real name of which 'rap'singer?":
        return "Shawn Carter is the real name of which 'rap' singer?"
    elif s in ["When sending an email you are given the option to 'Add Cc 'or' Add Bcc' . For which word is the 'B'the initial letter?", "When sending an email you are given the option to 'Add Cc' or' Add Bcc' . For which word is the 'B' the initial letter?"]:
        return "When sending an email you are given the option to 'Add Cc' or 'Add Bcc' . For which word is the 'B' the initial letter?"
    elif s in ["Covering an area in excess of 30,000 square miles, in which country can you find the' Dasht e Kavir Desert'the largest to lie wholly in it's own country?", "Covering an area in excess of 30,000 square miles, in which country can you find the' Dasht  e Kavir Desert' the largest to lie wholly in it's own country?"]:
        return "Covering an area in excess of 30,000 square miles, in which country can you find the 'Dasht e Kavir Desert' the largest to lie wholly in it's own country?"
    elif s == "In which year did South Korea hold its first' Grand Prix' motor race?":
        return "In which year did South Korea hold its first 'Grand Prix' motor race?"   

    return s


def has_answer(answers: list, text: str):
    for answer in answers:
        if answer.lower() in text.lower():
            return True
    return False


def poison(ctxs: list, new_answer: dict, old_answers: list, percent: float, poison_type: str):
    assert len(ctxs) == 100, len(ctxs)

    ids = [item["id"] for item in ctxs] # save this for later, so we can keep the order
    if "infobox-297352" in ids:
        breakpoint()

    # if there's no correct passages, no point in poisoning
    relevant_ctxs = [item for item in ctxs if has_answer(old_answers, item["title"] + " " + item["text"])]
    if not len(relevant_ctxs):
        return None

    non_relevant_ctxs = [item for item in ctxs if not has_answer(old_answers, item["title"] + " " + item["text"])]

    # sanity check
    for answer in old_answers:
        for item in non_relevant_ctxs:
            assert answer.lower() not in item["title"].lower(), f'{answer} found in {item["title"]}'
            assert answer.lower() not in item["text"].lower(), f'{answer} found in {item["text"]}'

    if poison_type == "top_percent":
        number_to_poison = max(1, round(percent * len(relevant_ctxs)))
        to_poison_idx = list(range(number_to_poison))
        to_poison = [item for i, item in enumerate(relevant_ctxs) if i in to_poison_idx]
        not_poisoned = [item for i, item in enumerate(relevant_ctxs) if i not in to_poison_idx]

    elif poison_type == "percent":
        to_poison_idx = random.sample(list(range(len(relevant_ctxs))), k=max(1, round(percent * len(relevant_ctxs))))
        to_poison = [item for i, item in enumerate(relevant_ctxs) if i in to_poison_idx]
        not_poisoned = [item for i, item in enumerate(relevant_ctxs) if i not in to_poison_idx]
        
    elif poison_type == "article":
        num_articles_to_poison = int(percent * 100)
        assert num_articles_to_poison >= 1
        titles = [item["title"] for i, item in enumerate(relevant_ctxs)]
        all_titles = list(dict.fromkeys(titles))
        titles_to_poison = all_titles[:num_articles_to_poison]
        to_poison = [item for item in relevant_ctxs if item["title"] in titles_to_poison]
        not_poisoned = [item for item in relevant_ctxs if item["title"] not in titles_to_poison]
        if percent == 1:
            assert num_articles_to_poison == 100
            assert len(not_poisoned) == 0

    assert len(to_poison) > 0
    # print(f"Poisoning {len(to_poison)} contexts")

    for item in non_relevant_ctxs + not_poisoned:
        item["poisoned"] = False

    final_ctxs = []
    all_possible_new_answers = [new_answer["text"]] + new_answer["aliases"] # NOTE: we don't use the aliases
    chosen_new_answer = random.sample(all_possible_new_answers, k=1)[0]

    # if an old answer is a subset of the chosen_new_answer it may have issues,
    #   by making it last, we can avoid issues with it expanding and then not replacing
    old_answers_has_new_inside = [has_answer([old_ans], chosen_new_answer) for old_ans in old_answers]
    old_answers = [x for _, x in sorted(zip(old_answers_has_new_inside, old_answers))]
    assert old_answers != [-1]

    if sum(old_answers_has_new_inside):
        flag_replace = True
    else:
        flag_replace = False

    for item in to_poison:
        item = copy.deepcopy(item)
        for old_answer in old_answers:
            new_title = ireplace(old_answer, chosen_new_answer, item["title"])
            item["title"] = new_title

            new_text = ireplace(old_answer, chosen_new_answer, item["text"])
            item["text"] = new_text

        item["poisoned"] = True
        item["hasanswer"] = False
        item["poisoned_answer"] = chosen_new_answer
        final_ctxs.append(item)

    assert len(final_ctxs + non_relevant_ctxs + not_poisoned) == 100, f"Should be 100, got: {len(final_ctxs + non_relevant_ctxs + not_poisoned)}"
    all_ctxs = final_ctxs + non_relevant_ctxs + not_poisoned
    all_ctxs_map = {item["id"]: item for item in all_ctxs}

    if percent == 1: # do a sanity check again
        for answer in old_answers:
            for item in all_ctxs:
                try: # the replacement would get flagged here instead
                    assert answer.lower() not in item["title"].lower(), f'{old_answers} found in {item["title"]}'
                    assert answer.lower() not in item["text"].lower(), f'{old_answers} found in {item["text"]}'
                except AssertionError:
                    if chosen_new_answer not in " ".join([item["title"], item["text"]]): # the new answer may be a subset
                        print(f'new {chosen_new_answer} not found in {item["title"] + " " + item["text"]}')
                        if not flag_replace:
                            raise AssertionError()



    return [all_ctxs_map[idv] for idv in ids] # need to be back in the original order

def create_conflicts(args):
    print(f"Given arguments={args}")

    # unfortunately, the data from conflicts has tokenization differences (due to NER and other models), so we have to resolve those
    dataset = "tqa" if "tqa" in args.path.lower() else "nq"
    with open(args.subproblems_path.replace(f"{dataset.lower()}_w_generations.json", f"{dataset.lower()}_{args.split}.json")) as fin:
        split_data = pd.DataFrame(json.load(fin))
        split_data.question = split_data.question.apply(lambda x: normalize(detokenize(x.split(" "))))

    conflicts_data = []
    with open(args.conflicts_path, "r") as fin:
        for i, line in enumerate(fin):
            if i == 0:
                continue
            else:
                conflicts_data.append(json.loads(line))

    with open(args.subproblems_path, "r") as fin:
        subproblems_data = json.load(fin)


    if args.model.lower() == "fid":
        with open(args.path, "r") as fin:
            data = json.load(fin)
        with open(args.subproblems_ctxs_path, "r") as fin:
            subproblems_ctxs_data = json.load(fin)
    elif args.model.lower() == "atlas":
        original_answers_data = []
        with open(f"data/{dataset.upper()}/test.json", "r") as fin:
            original_answers_data = json.load(fin, encoding="utf-8")
        original_answers_df = pd.DataFrame(original_answers_data)
        original_answers_df["norm_q"] = original_answers_df.question.apply(lambda x: str(manual_fix(normalize(x, dont_move_quotes=True))))
        q2ans = original_answers_df.set_index("norm_q").to_dict()["answers"]

        subproblems_ctxs_data = []
        with open(args.subproblems_ctxs_path, "r") as fin:
            for i, line in enumerate(fin):
                subproblems_ctxs_data.append(json.loads(line))
        data = []
        with open(args.path, "r") as fin:
            for i, line in enumerate(fin):
                data.append(json.loads(line))
    else:
        raise NotImplementedError(args.model.lower())


    def convert_to_ans(x: str, q2ans_dict):
        try:
            normed_q = manual_fix(" ".join(str(manual_fix(normalize(x, dont_move_quotes=True))).split()))
            return q2ans_dict[normed_q]
        except KeyError as e:
            print(normed_q)
            breakpoint()
            return None

    original_data = pd.DataFrame(data)
    if "passages" in original_data.columns: # atlas output has a different key
        original_data["ctxs"] = original_data["passages"]
    if "query" in original_data.columns: # atlas output has a different key
        original_data["question"] = original_data["query"].apply(lambda x: x.replace(" answer: <extra_id_0>", "").replace("question: ", "").strip())
        original_data["answers"] = original_data.question.apply(lambda x: convert_to_ans(x, q2ans))

    subproblems = pd.DataFrame(subproblems_data)
    subproblems.generated = subproblems.generated.apply(lambda x: split_generation(x)) # GPT-3 generated newlines, split on those
    subproblems.question = subproblems.question.apply(lambda x: str(manual_fix(normalize(x))))
    subproblem2q = {}
    for idx, df in subproblems.iterrows():
        for generated_q in df.generated:
            subproblem2q[generated_q] = df.question
    subproblem_qs = set(subproblems.question.tolist())


    # NOTE: conflicts data is a subset of the NQ dev set due to some not being easily substitutable
    conflicts = pd.DataFrame(conflicts_data)
    conflicts["question"] = conflicts["query"].apply(lambda x: normalize(detokenize(x.split(" "))))
    questions_in_conflicts = set(conflicts["question"].unique().tolist())
    print(f"## Length of the conflicts file is {len(questions_in_conflicts)}")

    original_data.question = original_data.question.apply(lambda x: manual_fix(normalize(detokenize(x.split(" ")))))
    original_data = original_data[original_data.question.apply(lambda x: x in questions_in_conflicts)]
    set_of_original_questions = set(original_data.question.tolist())

    # not sure why the conflicts qs has two more than the others ... but ignore them
    # for references, those are {'birth certificates are most often completed by the department', 'the agreement over how states would be represented in congress was known as'}
    equal = 2 if dataset.lower() == "nq" else 0
    assert len(questions_in_conflicts - set_of_original_questions) == equal, len(questions_in_conflicts - set_of_original_questions)
    assert len(questions_in_conflicts - subproblem_qs) == equal, len(questions_in_conflicts - subproblem_qs)

    # verify subproblem ctxs
    subproblems_ctxs = pd.DataFrame(subproblems_ctxs_data)
    if "passages" in subproblems_ctxs.columns: # atlas output has a different key
        subproblems_ctxs["ctxs"] = subproblems_ctxs["passages"]
    if "query" in subproblems_ctxs.columns: # atlas output has a different key
        subproblems_ctxs["question"] = subproblems_ctxs["query"].apply(lambda x: x.replace(" answer: <extra_id_0>", "").replace("question: ", "").strip())

    # verify the link between subproblem and original q
    for idx, subq in enumerate(subproblems_ctxs.question.tolist()):
        try:
            assert subq in subproblem2q, subq
        except AssertionError as e:
            # somehow a few got an extra ' in preprocessing, fix them here
            if subq == "What's the time?": 
                subproblems_ctxs.question[idx] = 'Whats the time?'
                assert subproblems_ctxs.question[idx] in subproblem2q
            elif subq == "I'm a little teapot, short and stout. Here is my handle, here is my spout.":
                subproblems_ctxs.question[idx] = "Im a little teapot, short and stout. Here is my handle, here is my spout."
                assert subproblems_ctxs.question[idx] in subproblem2q
            else:
                raise e
    
    # verify the reverse link
    set_of_subproblems_with_ctx = subproblems_ctxs.question.tolist()
    for idx, subq_list in enumerate(subproblems.generated.tolist()):
        for generated_q in subq_list:
            assert generated_q in set_of_subproblems_with_ctx


    # do some validation
    subproblems = subproblems[subproblems.question.isin(questions_in_conflicts)]
    original_data = original_data[original_data.question.isin(questions_in_conflicts)]
    og_len = len(original_data)
    subproblems.question = subproblems.question.astype(str)
    original_data.question = original_data.question.astype(str)

    # merge with generated GPT-3 paraphrases and sanity check
    original_data = original_data.merge(subproblems, on='question', how='inner')
    assert og_len == len(original_data)
    assert original_data.generated.apply(lambda x: pd.isnull(x).sum()).sum() == 0

    # merge with new contexts and sanity check
    conflicts_chosen = []
    for q, df in conflicts.groupby("question"):
        conflicts_chosen.append(df.iloc[[0]])

    conflicts_only = pd.concat(conflicts_chosen, axis=0)
    original_data = original_data.merge(conflicts_only, on='question', how='inner')
    assert og_len == len(original_data)
    assert original_data.is_substitute.isnull().sum() == 0

    # Keep only the split
    before = len(original_data)
    original_data = original_data[original_data.question.apply(lambda x: x in split_data.question.tolist())]
    print(f"Changing from {before} to {len(original_data)} after split check")

    # go through each of the ctxs and change them
    original_data["ctxs_poisoned"] = original_data.apply(lambda x: poison(x["ctxs"], x["gold_answers"][0], x["answers"], args.percent, args.poison_type), axis=1)
    # some of them have no relevant passages
    original_data = original_data[original_data["ctxs_poisoned"].apply(lambda x: x is not None)]
    print(f"## Length of the data file after removing instances with zero correct passages is {len(original_data)}")

    ### Generate FiD inference files: question, answers, ctxs ###
    # for measuring difference in retrieval
    dist_of_diffs_max = []
    dist_of_diffs_median = []
    dist_of_diffs_mean = []
    dist_of_diffs_min = []
    dist_of_diffs = []

    all_instances = []
    original_instances = []
    num_generated_qs_per_q = []
    num_generated_qs_w_context_per_q = []
    ctx_name = "ctxs" if args.model.lower() == "fid" else "passages" # need different keys for two-stage atlas model
    for q, df in tqdm.tqdm(original_data.groupby("question")):
        assert len(df) == 1
        num_diff = []
        df = df.iloc[0]
        poisoned_map = {int(item["id"].replace("infobox-", "").replace("wiki:", "")): item for item in df.ctxs_poisoned if item["poisoned"]}
        original_passage_ids = [int(item["id"].replace("infobox-", "").replace("wiki:", "")) for item in df.ctxs]
        assert len(poisoned_map), poisoned_map
        fake_answers = [item["poisoned_answer"] for item in df.ctxs_poisoned if item["poisoned"]]
        original_poisoned_item = {
            "question": q,
            "answers": df.answers,
            ctx_name: df.ctxs_poisoned,
            "original_q": q,
            "fake_answer": fake_answers,
            "setting": "original_q_poisoned_ctxs",
        }


        all_instances.append(original_poisoned_item)
        original_item = {
            "question": q,
            "answers": df.answers,
            ctx_name: df.ctxs,
            "original_q": q,
            "fake_answer": [],
            "setting": "original_ctxs",
        }
        original_instances.append(original_item)
        num_generated_qs_per_q.append(len(df.generated))
        num_generated_qs_w_context = 0
        for generated_q in df.generated:
            # two different settings: (1) new question same poisoned contexts (2) new question, new passages

            # (1) just swap out the main question
            generated_q_original_ctxs = {
                "question": generated_q,
                "answers": df.answers,
                ctx_name: df.ctxs_poisoned,
                "original_q": q,
                "fake_answer": fake_answers,    
                "setting": "generated_q_poisoned_ctxs",
            }
            all_instances.append(generated_q_original_ctxs)
        
            # (2) make sure we propogate the poisoned ctxs
            new_retrieved_ctxs = subproblems_ctxs[subproblems_ctxs.question == generated_q].ctxs
            
            if len(new_retrieved_ctxs) > 1: # sometimes there is question overlap, but ctxs are the same given the same q
                new_retrieved_ctxs = new_retrieved_ctxs.iloc[0]
            else:
                new_retrieved_ctxs = new_retrieved_ctxs.iloc[0]

            final_retrieved_ctxs = []
            fake_answers_subproblem = []
            replaced = 0
            for ctx in new_retrieved_ctxs:
                int_id = int(ctx["id"].replace("infobox-", "").replace("wiki:", ""))
                if int_id in poisoned_map:
                    replaced += 1
                    final_retrieved_ctxs.append(poisoned_map[int_id])
                    fake_answers_subproblem.append(poisoned_map[int_id]["poisoned_answer"])
                else:
                    final_retrieved_ctxs.append(ctx)
            new_passage = set([int(item["id"].replace("infobox-", "").replace("wiki:", "")) for item in new_retrieved_ctxs]) - set(original_passage_ids)
            # print(f"Replaced {replaced}/{len(poisoned_map)} with {len(new_passage)} new passages from the augmented query with max new: {max(new_passage)} and old {max(original_passage_ids)}")
            num_diff.append(len(new_passage))

            generated_q_new_ctxs = {
                "question": generated_q,
                "answers": df.answers,
                ctx_name: final_retrieved_ctxs,
                "original_q": q,
                "fake_answer": fake_answers_subproblem,
                "setting": "generated_q_new_ctxs",
            }
            all_instances.append(generated_q_new_ctxs)

            # (3) Original Question with new contexts
            og_q_new_ctxs = {
                "question": q,
                "answers": df.answers,
                ctx_name: final_retrieved_ctxs,
                "original_q": q,
                "fake_answer": fake_answers_subproblem,
                "setting": "og_q_new_ctxs",
            }
            all_instances.append(og_q_new_ctxs)
            num_generated_qs_w_context += 1
        num_generated_qs_w_context_per_q.append(num_generated_qs_w_context)
        dist_of_diffs_max.append(np.max(num_diff))
        dist_of_diffs_median.append(np.median(num_diff))
        dist_of_diffs_mean.append(np.mean(num_diff))
        dist_of_diffs_min.append(np.min(num_diff))
        dist_of_diffs.extend(num_diff)

    
    results_folder = f"results_{dataset}_{args.split}/{args.model}/"
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    list_of_results_list = [dist_of_diffs_max, dist_of_diffs_median, dist_of_diffs_mean, dist_of_diffs_min, dist_of_diffs]
    for name, list_of_results in zip(["max", "median", "mean", "min", "all"], list_of_results_list):
        a4_dims = (4, 2)
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.histplot(list_of_results, ax=ax, bins=20)
        plt.ylabel("Count")
        plt.xlabel("# of New Passages in Augmented Questions")
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f"dist_{name}_diff.png"))
        plt.savefig(os.path.join(results_folder, f"dist_{name}_diff.pdf"))
        plt.close()


    # make folder if it doesn't exist
    output_folder = "/".join(args.output_file.split("/")[:-1])
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if args.model.lower() == "atlas": # jsonl format
        print(f"Writing out data to {args.output_file} with length {len(all_instances)} with an average of {np.mean(num_generated_qs_per_q)} new qs per q and {np.mean(num_generated_qs_w_context_per_q)} with context")  
        with open(args.output_file, "w") as fout:
            for inst in all_instances:
                fout.write(json.dumps(inst) + "\n")

        print(f"Writing out original data with length {len(original_instances)}")  
        with open(args.output_file.split("conflicts")[0] + "original-0.json", "w") as fout:
            for inst in original_instances:
                fout.write(json.dumps(inst) + "\n")

    elif args.model.lower() == "fid": # json format
        print(f"Writing out data to {args.output_file} with length {len(all_instances)} with an average of {np.mean(num_generated_qs_per_q)} new qs per q and {np.mean(num_generated_qs_w_context_per_q)} with context")  
        with open(args.output_file, "w") as fout:
            json.dump(all_instances, fout, indent=4)

        print(f"Writing out original data with length {len(original_instances)}")  
        with open(args.output_file.split("conflicts")[0] + "original-0.json", "w") as fout:
            json.dump(original_instances, fout, indent=4)
    else:
        raise NotImplementedError(args.model.lower())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to file containing original NQ data with DPR results', type=str, required=True)
    parser.add_argument('-c', '--conflicts_path', help='path to file containing conflicting data (from Apple paper)', type=str, required=True)
    parser.add_argument('--subproblems_ctxs_path', help='path to file containing subproblems data for ctxs (from DPR)', type=str, required=True)
    parser.add_argument('-s', '--subproblems_path', help='path to file containing subproblems data (auto generated paraphrases)', type=str, required=True)
    parser.add_argument('--percent', help='what percent of correct passages with the answer to replace with the fake entity', type=float, default=0.5)
    parser.add_argument('--poison_type', help='whether to poison n percent, top n percent or per articles', type=str, default="percent")
    parser.add_argument('-o', '--output_file', help='path to file to write data to', type=str, required=True)
    parser.add_argument('--split', help='what split the data should be (dev/test)', type=str, required=True)
    parser.add_argument('--model', help='what model format to generate for (fid, atlas)', type=str, default="fid")
    args = parser.parse_args()
    create_conflicts(args)

    # Example command
    # python3 aa/create_conflicts.py -p artifacts/test.json -o artifacts/nq-dev-w-conflicts.json -c ml-knowledge-conflicts/datasets/substitution-sets/MRQANaturalQuestionsDevType.jsonl -s data/nq_w_generations.json --subproblems_ctxs_path /artifacts/gpt-3-search.json --poison_type article

    # python3 aa/create_conflicts.py -p artifacts/test.json -o artifacts/tqa-dev-w-conflicts.json -c ml-knowledge-conflicts/datasets/substitution-sets/MRQANaturalQuestionsDevType.jsonl -s data/tqa_w_generations.json --subproblems_ctxs_path artifacts/gpt-3-search.json --poison_type article
