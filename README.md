# Defending Against Disinformation Attacks in Open-Domain Question Answering

This is the official repository for the paper [Defending Against Disinformation Attacks in Open-Domain Question Answering](https://arxiv.org/abs/2212.10002).

## Overview

This paper proposes to defend against disinformation poisoning attacks in open-domain question answering (e.g. e.g. someone malicously puts a website with fake information to be indexed by search engines). We provide tools and data for generating augmented queries and evaluating the robustness of question answering models with various methods.

## Quick Start

Most of the data is pre-generated and available in a Hugging Face dataset: [orionweller/Defending-Agaisnt-Disinformation-EACL-24](https://huggingface.co/datasets/orionweller/Defending-Agaisnt-Disinformation-EACL-24).

### Initial Setup

1. Install requirements:
   ```
   conda env create --file conda_env.yml
   conda activate conflicts
   ```

2. Clone the pre-generated data:
   ```
   git clone https://huggingface.co/datasets/orionweller/Defending-Agaisnt-Disinformation-EACL-24
   mv Defending-Agaisnt-Disinformation-EACL-24/* .
   ```

## Detailed Instructions

<details>
<summary><strong>Generate Augmented Queries</strong></summary>

Pre-generated augmented queries can be found in `data/*/*_w_generations*.json`.

To regenerate:

1. Get questions for GPT-3:
   ```
   python get_questions_from_dataset.py
   ```

2. Run GPT-3 paraphrasing:
   ```
   python prompt_gpt3.py --dataset_name {nq,tqa} --API_TOKEN <YOUR_API_TOKEN>
   ```

3. Run LLama-2 paraphrasing:
   ```
   python prompt_llama2.py
   ```

Note: GPT-3 Davinci may be unavailable. Pre-generated questions are located in (using TQA as an example):
- `data/TQA/tqa_w_generations.json` (GPT-3)
- `data/TQA/tqa_w_generations_llama.json` (Llama2)

</details>

<details>
<summary><strong>Generate Disinformation Conflicts</strong></summary>

To recreate the conflicting data:

1. Clone the knowledge conflicts repository:
   ```
   git clone https://github.com/apple/ml-knowledge-conflicts.git
   cd ml-knowledge-conflicts
   ```

2. Follow setup instructions:
   ```
   bash setup.sh
   ```

3. Generate substitutions for Natural Questions:
   ```
   PYTHONPATH=. python src/load_dataset.py -d MRQANaturalQuestionsDev -w wikidata/entity_info.json.gz
   PYTHONPATH=. python src/generate_substitutions.py --inpath datasets/normalized/MRQANaturalQuestionsDev.jsonl --outpath datasets/substitution-sets/MRQANaturalQuestionsDevType.jsonl corpus-substitution
   ```

4. Generate substitutions for TriviaQA:
   ```
   PYTHONPATH=. python src/load_dataset.py -d MRQATriviaQADev -w wikidata/entity_info.json.gz
   PYTHONPATH=. python src/generate_substitutions.py --inpath datasets/normalized/MRQATriviaQADev.jsonl --outpath datasets/substitution-sets/MRQATriviaQADevType.jsonl corpus-substitution
   ```

</details>

<details>
<summary><strong>Search and Predict Using ATLAS and FiD</strong></summary>

### Download Models

1. Download FiD models:
   ```
   bash scripts/download_FiD_models.sh
   ```

2. Set up FiD dataset:
   - Follow instructions in the [FiD repo](https://github.com/facebookresearch/FiD)
   - Run `get-data.sh` and copy data to `data/`

3. Set up ATLAS:
   - Clone the [ATLAS repo](https://github.com/facebookresearch/atlas)
   - Download ATLAS files:
     ```
     bash scripts/download_atlas_models.sh
     ```

4. Generate embedding indices:
   ```
   bash scripts/generate_all_embeddings.sh {nq,tqa}
   ```

### Gather Model Retrieval and Predictions

#### FiD

1. Prepare retrieval data:
   ```
   python convert_generations_to_dpr_format.py -p data/NQ/nq_w_generations.json -o artifacts/questions_to_retrieve_nq.json
   ```

2. Retrieve data:
   ```
   qsub -N ret-fid retrieve_FiD.sh {nq,tqa}
   ```

3. Create poisoned data:
   ```
   bash bulk_create_conflicts.sh
   ```

4. Run FiD evaluation:
   ```
   bash evaluate_all_FiD.sh
   ```

#### ATLAS

1. Convert data format:
   ```
   python convert_generations_to_dpr_format.py -r -p data/NQ/nq_w_generations.json -o artifacts/questions_to_retrieve_nq_atlas.json
   ```

2. Retrieve passages:
   ```
   qsub -N ret-atlas scripts/retrieve_atlas.sh {nq,tqa}
   ```

3. Create poisoned data:
   ```
   bash bulk_create_conflicts.sh
   ```

4. Run ATLAS evaluation:
   ```
   bash scripts/evaluate_all_ATLAS.sh
   ```

</details>

<details>
<summary><strong>Evaluation</strong></summary>

1. Calculate results:
   ```
   bash scripts/bulk_calculate_results.sh
   ```

2. Analyze overall groups:
   ```
   python3 collect_results_across_percents.py -r results_nq_dev
   ```

</details>

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@inproceedings{weller-etal-2024-defending,
    title = "Defending Against Disinformation Attacks in Open-Domain Question Answering",
    author = "Weller, Orion  and
      Khan, Aleem  and
      Weir, Nathaniel  and
      Lawrie, Dawn  and
      Van Durme, Benjamin",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-short.35",
    pages = "402--417",
}
```