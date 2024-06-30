import os
import pandas as pd
import json
from datasets import load_dataset
import random
import string

from create_conflicts import detokenize, manual_fix, normalize

random.seed(42)


##### NQ #####

nq_with_conflicts = []
with open("ml-knowledge-conflicts/datasets/substitution-sets/MRQANaturalQuestionsDevType.jsonl", "r") as fin:
    for idx, line in enumerate(fin):
        if not idx: # first is title of dataset
            continue
        nq_with_conflicts.append({"question": normalize(detokenize(json.loads(line)["query"].split(" ")))})
nq_with_conflicts_set = set([item["question"].lower() for item in nq_with_conflicts])


questions = []
dataset = load_dataset("nq_open")
for item in dataset["validation"]:
    # breakpoint()
    # TODO match substutions here, will make life easier
    q = manual_fix(normalize(detokenize(item["question"].split(" "))))
    if q in nq_with_conflicts_set:
        questions.append({"question": item["question"]})


print(f"NQ: From {len(nq_with_conflicts_set)} conflicts we get {len(questions)} qs")
assert len(nq_with_conflicts_set) - len(questions) == 2

indices_dev = random.sample(list(range(len(questions))), k=round(0.5 * len(questions)))


with open("data/NQ/nq_dev.json", "w") as fout:
    json.dump([item for i, item in enumerate(questions) if i in indices_dev], fout, indent=4)

with open("data/NQ/nq_test.json", "w") as fout:
    json.dump([item for i, item in enumerate(questions) if i not in indices_dev], fout, indent=4)

with open("data/NQ/nq.json", "w") as fout:
    json.dump(questions, fout, indent=4)

##### TQA #####

tqa_with_conflicts = []
with open("ml-knowledge-conflicts/datasets/substitution-sets/MRQATriviaQADevType.jsonl", "r") as fin:
    for idx, line in enumerate(fin):
        if not idx:
            continue
        tqa_with_conflicts.append({"question": normalize(detokenize(json.loads(line)["query"].split(" ")))})
tqa_with_conflicts_set = set([item["question"] for item in tqa_with_conflicts])

tqa_dev_qs = []
with open("data/TQA/test.json", "r") as fin: # MRC workshops dev is FiD test
    tqa_dev = json.load(fin)
for line in tqa_dev:
    q = normalize(detokenize(line["question"].split(" ")))
    if q in tqa_with_conflicts_set:
        tqa_dev_qs.append({"question": line["question"]})

print(f"TQA: from {len(tqa_with_conflicts_set)} conflicts we get {len(tqa_dev_qs)} qs out of")
assert len(tqa_with_conflicts_set) == len(tqa_dev_qs)

indices_dev = random.sample(list(range(len(tqa_dev_qs))), k=round(0.5 * len(tqa_dev_qs)))

with open("data/TQA/tqa_dev.json", "w") as fout:
    json.dump([item for i, item in enumerate(tqa_dev_qs) if i in indices_dev], fout, indent=4)

with open("data/TQA/tqa_test.json", "w") as fout:
    json.dump([item for i, item in enumerate(tqa_dev_qs) if i not in indices_dev], fout, indent=4)

with open("data/TQA/tqa.json", "w") as fout:
    json.dump(tqa_dev_qs, fout, indent=4)



