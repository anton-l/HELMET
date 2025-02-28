import os
import json
import numpy as np
import pandas as pd
import yaml
import argparse
from dataclasses import dataclass, asdict
from tqdm import tqdm

dataset_to_metrics = {
    "json_kv": "substring_exact_match",
    "nq": "substring_exact_match",
    "popqa": "substring_exact_match",
    "triviaqa": "substring_exact_match",
    "hotpotqa": "substring_exact_match",

    "narrativeqa": ["gpt-4-score", ],
    "msmarco_rerank_psg": "NDCG@10",

    "trec_coarse": "exact_match",
    "trec_fine": "exact_match",
    "banking77": "exact_match",
    "clinic150": "exact_match",
    "nlu": "exact_match",

    "qmsum": "rougeL_recall",
    "multi_lexsum": ["gpt4-f1"],

    "ruler_niah_s_1": "ruler_recall",
    "ruler_niah_s_2": "ruler_recall",
    "ruler_niah_s_3": "ruler_recall",
    "ruler_niah_mk_1": "ruler_recall",
    "ruler_niah_mk_2": "ruler_recall",
    "ruler_niah_mk_3": "ruler_recall",
    "ruler_niah_mq": "ruler_recall",
    "ruler_niah_mv": "ruler_recall",
    "ruler_fwe": "ruler_recall",
    "ruler_cwe": "ruler_recall",
    "ruler_vt": "ruler_recall",
    "ruler_qa_1": "substring_exact_match",
    "ruler_qa_2": "substring_exact_match",

    "infbench_qa": ["rougeL_f1"],
    "infbench_choice": ["exact_match"],
    "infbench_sum": ["gpt4-f1"],

    "alce_asqa": ["str_em", "citation_rec", "citation_prec"],
    "alce_qampari": ["qampari_rec_top5", "citation_rec", "citation_prec"],
}
dataset_to_metrics = {k: [v] if isinstance(v, str) else v for k, v in dataset_to_metrics.items()}
custom_avgs = {
    "Recall": ["json_kv substring_exact_match", "ruler_niah_mk_2 ruler_recall", "ruler_niah_mk_3 ruler_recall",
               "ruler_niah_mv ruler_recall"],
    "RAG": ['nq substring_exact_match', 'hotpotqa substring_exact_match', 'popqa substring_exact_match',
            'triviaqa substring_exact_match', ],
    "ICL": ['trec_coarse exact_match', 'trec_fine exact_match', 'banking77 exact_match', 'clinic150 exact_match',
            'nlu exact_match'],
    "Cite": ['alce_asqa str_em', 'alce_asqa citation_rec', 'alce_asqa citation_prec', 'alce_qampari qampari_rec_top5',
             'alce_qampari citation_rec', 'alce_qampari citation_prec', ],
    "Re-rank": ['msmarco_rerank_psg NDCG@10', ],
    "LongQA": ['narrativeqa gpt-4-score', 'infbench_qa rougeL_f1', 'infbench_choice exact_match', ],
    "Summ": ['infbench_sum gpt4-f1', 'multi_lexsum gpt4-f1', ],
    "RULER": ['ruler_niah_s_1 ruler_recall', 'ruler_niah_s_2 ruler_recall', 'ruler_niah_s_3 ruler_recall',
              'ruler_niah_mk_1 ruler_recall', 'ruler_niah_mk_2 ruler_recall', 'ruler_niah_mk_3 ruler_recall',
              'ruler_niah_mq ruler_recall', 'ruler_niah_mv ruler_recall', 'ruler_cwe ruler_recall',
              'ruler_fwe ruler_recall', 'ruler_vt ruler_recall', 'ruler_qa_1 substring_exact_match',
              'ruler_qa_2 substring_exact_match'],
    "Ours-Real": ['RAG', 'ICL', 'Cite', 'Re-rank', 'LongQA', 'Summ'],
    "Ours": ['Recall', 'RAG', 'ICL', 'Cite', 'Re-rank', 'LongQA', 'Summ'],
}


@dataclass
class arguments:
    tag: str = "v1"
    input_max_length: int = 131072
    generation_max_length: int = 100
    generation_min_length: int = 0
    max_test_samples: int = 100
    shots: int = 2
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    use_chat_template: bool = False
    seed: int = 42
    num_depths: int = 11
    test_name: str = ""
    dataset: str = "nq"
    output_dir: str = "output"
    popularity_threshold: float = 3
    flenqa_ctx_size: int = 1000

    category: str = "synthetic"

    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_path(self):
        tag = self.tag
        if "flenqa" in self.dataset:
            tag += f"_ctx{self.flenqa_ctx_size}"
        path = os.path.join(self.output_dir,
                            "{args.dataset}_{tag}_{args.test_name}_in{args.input_max_length}_size{args.max_test_samples}_shots{args.shots}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json".format(
                                args=self, tag=tag))

        if os.path.exists(path.replace(".json", "-gpt4eval_o.json")):
            return path.replace(".json", "-gpt4eval_o.json")
        if "alce" in self.dataset:
            return path.replace(".json", ".json.score")

        if os.path.exists(path + ".score"):
            return path + ".score"
        return path

    def get_metric_name(self):
        for d, m in dataset_to_metrics.items():
            if d in self.dataset:
                return d, m
        return None

    def get_averaged_metric(self):
        path = self.get_path()
        print(path)
        if not os.path.exists(path):
            print("path doesn't exist")
            return None
        with open(path) as f:
            results = json.load(f)

        _, metric = self.get_metric_name()
        if path.endswith(".score"):
            if any([m not in results for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results[m] for m in metric}
        else:
            if any([m not in results["averaged_metrics"] for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results['averaged_metrics'][m] for m in metric}

        s = {m: v * (100 if m == "gpt4-f1" else 1) * (100 / 3 if m == "gpt-4-score" else 1) for m, v in s.items()}
        print("found scores:", s)
        return s


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Collect evaluation results for language models')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save the results CSV file')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model being evaluated')
    parser.add_argument('--context_length', type=int, required=True,
                        help='Context length to evaluate (e.g., 8192, 16384, 32768)')
    parser.add_argument('--use_chat_template', type=bool, default=False,
                        help='Whether to use chat template for the model')


    args = parser.parse_args()

    # Set variables from arguments
    output_file = args.output_file
    model_name = args.model_name
    MAX_LENGTH = args.context_length
    use_chat_template = args.use_chat_template

    print(f"Evaluating model: {model_name}")
    print(f"Context length: {MAX_LENGTH}")
    print(f"Use chat template: {use_chat_template}")
    print(f"Results will be saved to: {output_file}")

    # Define model configuration
    models_configs = [
        {"model": model_name, "training_length": MAX_LENGTH, "use_chat_template": use_chat_template}
    ]

    if MAX_LENGTH == 8192:
        config_dir = "configs/configs_8k"
    elif MAX_LENGTH == 16384:
        config_dir = "configs/configs_16k"
    elif MAX_LENGTH == 32768:
        config_dir = "configs/configs_32k"
    else:
        raise ValueError(f"No configs for max length {MAX_LENGTH}")

    configs = ["recall.yaml", "icl.yaml", "longqa.yaml", "rag.yaml", "rerank.yaml"]
    configs = [f"{config_dir}/{c}" for c in configs]

    datasets_configs = []
    for config in configs:
        try:
            c = yaml.safe_load(open(config))
            print(f"Loaded config {config}")

            if isinstance(c["generation_max_length"], int):
                c["generation_max_length"] = ",".join([str(c["generation_max_length"])] * len(c["datasets"].split(",")))
            if isinstance(c["input_max_length"], int):
                c["input_max_length"] = ",".join([str(c["input_max_length"])] * len(c["datasets"].split(",")))

            for d, t, l, g in zip(c['datasets'].split(','), c['test_files'].split(','),
                                  c['input_max_length'].split(','), c['generation_max_length'].split(',')):
                datasets_configs.append(
                    {"dataset": d, "test_name": os.path.basename(os.path.splitext(t)[0]),
                     "input_max_length": int(l), "generation_max_length": int(g),
                     "use_chat_template": c["use_chat_template"],
                     "max_test_samples": c["max_test_samples"], 'shots': c['shots']})
        except Exception as e:
            print(f"Error loading config {config}: {e}")

    df = []
    for model in tqdm(models_configs):
        if model["use_chat_template"]:
            del model["use_chat_template"]
        args = arguments()
        args.tag = "v1"
        args.output_dir = f"output/{model['model']}"

        for dataset in datasets_configs:
            args.update(dataset)
            args.update(model)

            metric = args.get_averaged_metric()
            dsimple, mnames = args.get_metric_name()

            if metric is None:
                continue

            for k, m in metric.items():
                df.append({**asdict(args), **model,
                           "metric name": k, "metric": m,
                           "dataset_simple": dsimple + " " + k,
                           "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}"
                           })

    all_df = pd.DataFrame(df)
    print(f"Collected {len(all_df)} results")

    lf_df = all_df.pivot_table(index=["model"], columns="dataset_simple", values="metric", sort=False)
    lf_df = lf_df.reset_index()

    # Calculate aggregate metrics
    custom_avgs = {
        "Recall": ["json_kv substring_exact_match", "ruler_niah_mk_2 ruler_recall", "ruler_niah_mk_3 ruler_recall",
                   "ruler_niah_mv ruler_recall"],
        "ICL": ['trec_coarse exact_match', 'trec_fine exact_match', 'banking77 exact_match', 'clinic150 exact_match',
                'nlu exact_match'],
        "LongQA": ['infbench_qa rougeL_f1', 'infbench_choice exact_match', ],
        "RAG": ['nq substring_exact_match', 'hotpotqa substring_exact_match', 'popqa substring_exact_match',
                'triviaqa substring_exact_match', ],
        "Re-rank": ['msmarco_rerank_psg NDCG@10', ],
        "Average-Real": ['RAG', 'ICL', 'Re-rank', 'LongQA'],
        "Average-All": ['Recall', 'RAG', 'ICL', 'Re-rank', 'LongQA'],
    }

    for category, columns in custom_avgs.items():
        if category in ["Average-Real", "Average-All"]:
            available_columns = [col for col in columns if any(c in lf_df.columns for c in custom_avgs.get(col, []))]
            if available_columns:
                available_values = []
                for col in available_columns:
                    col_columns = [c for c in custom_avgs.get(col, []) if c in lf_df.columns]
                    if col_columns:
                        lf_df[col] = np.mean(lf_df[col_columns], axis=1)
                        available_values.append(col)
                if available_values:
                    lf_df[category] = np.mean(lf_df[available_values], axis=1)
        else:
            available_columns = [c for c in columns if c in lf_df.columns]
            if available_columns:
                lf_df[category] = np.mean(lf_df[available_columns], axis=1)

    cols = ["model", "Average-Real", "Average-All", "Recall", "ICL", "LongQA", "RAG", "Re-rank"]
    final_cols = []
    for c in cols:
        if c in lf_df.columns:
            final_cols.append(c)
    for c in lf_df.columns:
        if c not in final_cols:
            final_cols.append(c)

    lf_df = lf_df[final_cols]

    print(f"Results for {model_name} with context length {MAX_LENGTH}:")

    lf_df.to_csv(output_file, index=False)
    print(f"Results written to {output_file}")

    print("\nSummary of results:")
    for col in ['Average-Real', 'Average-All', 'Recall', 'ICL', 'LongQA', 'RAG', 'Re-rank']:
        if col in lf_df.columns:
            print(f"{col}: {lf_df[col].values[0]:.2f}")