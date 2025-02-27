import os
import json
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, asdict
from tqdm import tqdm
from transformers.agents.python_interpreter import MAX_LEN_OUTPUT

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

    def get_metric_by_depth(self):
        path = self.get_path()
        path = path.replace(".score", '')
        print(path)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            results = json.load(f)

        output = []
        _, metric = self.get_metric_name()
        metric = metric[0]
        keys = ["depth", "k", metric]
        for d in results["data"]:
            o = {}
            for key in keys:
                if key == "k" and "ctxs" in d:
                    d["k"] = len(d['ctxs'])
                if key not in d:
                    print("no", key)
                    return None
                o[key] = d[key]
            o["metric"] = o.pop(metric)
            output.append(o)

        df = pd.DataFrame(output)
        dfs = df.groupby(list(output[0].keys())[:-1]).mean().reset_index()

        return dfs.to_dict("records")


if __name__ == "__main__":
    # comment out the models you don't want to include
    models_configs = [
        # {"model": "Llama-3.2-1B", "training_length": 131072, "use_chat_template": False},
        # {"model": "Llama-3.2-1B-Instruct", "training_length": 131072},
        # {"model": "Qwen2.5-1.5B", "training_length": 131072, "use_chat_template": False},
        # {"model": "Qwen2.5-1.5B-Instruct", "training_length": 131072},
        # {"model": "SmolLM2-1.7B", "training_length": 8192, "use_chat_template": False},
        # {"model": "SmolLM2-1.7B-Instruct-rope300k-reup", "training_length": 16384},
        # {"model": "SmolLM2-Instruct-16k-SeaLong-ST5k-rope300k", "training_length": 16384},
        # {"model": "SmolLM2-Instruct-16k-SeaLong-LongAlign-ST5k-rope300k", "training_length": 16384},
        #
        # {"model": "SmolLM2-16k-SmolTalk1M-AceMath500k-CodeIO-ep2-8k", "training_length": 8192, "use_chat_template": False},
        # {"model": "SmolLM2-16k-SmolTalk1M-AceMath500k-CodeIO-ep1-8k", "training_length": 8192, "use_chat_template": False},
        # {"model": "SmolLM2-16k-SmolTalk1M-AceMath500k-ep1-8k", "training_length": 8192, "use_chat_template": False},
        # {"model": "SmolLM2-16k-SmolTalk1M-AceMath100k-CodeIO-ep1-rope300k", "training_length": 16384, "use_chat_template": False},
        # {"model": "SmolLM2-16k-SmolTalk1M-AceMath100k-CodeIO-ep1-rope100k", "training_length": 16384, "use_chat_template": False},

        # {"model": "SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k", "training_length": 16384, "use_chat_template": True},
        # {"model": "SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k-1e-4-ep2", "training_length": 16384, "use_chat_template": True},
        # {"model": "SmolLM2-Ins-16k-SeaLong-LongAlign-ST5k-v2-rope300k-1e-4", "training_length": 16384, "use_chat_template": False},
        # {"model": "SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope500k-1e-4", "training_length": 16384, "use_chat_template": True},

        # #   "HuggingFaceTB/SmolLM2-1.7B-Instruct-rope300k-reup"
        # {"model": "SmolLM2-1.7B-Instruct-rope300k-reup", "training_length": 16384},
        # #   "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-ST5k-rope300k"
        # {"model": "SmolLM2-Instruct-16k-SeaLong-ST5k-rope300k", "training_length": 16384},
        # #   "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST5k-rope300k"
        # {"model": "SmolLM2-Instruct-16k-SeaLong-LongAlign-ST5k-rope300k", "training_length": 16384},
        # #   "HuggingFaceTB/SmolLM2-16k-SmolTalk1M-AceMath500k-CodeIO-ep2-8k"
        # {"model": "SmolLM2-16k-SmolTalk1M-AceMath500k-CodeIO-ep2-8k", "training_length": 8192},
        # #   "HuggingFaceTB/SmolLM2-16k-SmolTalk1M-AceMath500k-CodeIO-ep1-8k"
        # {"model": "SmolLM2-16k-SmolTalk1M-AceMath500k-CodeIO-ep1-8k", "training_length": 8192},
        # #   "HuggingFaceTB/SmolLM2-16k-SmolTalk1M-AceMath500k-ep1-8k"
        # {"model": "SmolLM2-16k-SmolTalk1M-AceMath500k-ep1-8k", "training_length": 8192},
        # #   "HuggingFaceTB/SmolLM2-16k-SmolTalk1M-AceMath100k-CodeIO-ep1-rope300k"
        # {"model": "SmolLM2-16k-SmolTalk1M-AceMath100k-CodeIO-ep1-rope300k", "training_length": 16384},
        # #   "HuggingFaceTB/SmolLM2-16k-SmolTalk1M-AceMath100k-CodeIO-ep1-rope100k"
        # {"model": "SmolLM2-16k-SmolTalk1M-AceMath100k-CodeIO-ep1-rope100k", "training_length": 16384},
        # #   "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k"
        # {"model": "SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k", "training_length": 16384},
        # #   "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k-1e-4-ep2"
        # {"model": "SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k-1e-4-ep2", "training_length": 16384},
        # #   "HuggingFaceTB/SmolLM2-Ins-16k-SeaLong-LongAlign-ST5k-v2-rope300k-1e-4"
        # {"model": "SmolLM2-Ins-16k-SeaLong-LongAlign-ST5k-v2-rope300k-1e-4", "training_length": 16384},
        # #   "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope500k-1e-4"
        # {"model": "SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope500k-1e-4", "training_length": 16384},

        # #  "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope500k-5e-5-ep2"
        # {"model": "SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope500k-5e-5-ep2", "training_length": 16384},
        # #  "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k-1e4-2"
        # {"model": "SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k-1e4-2", "training_length": 16384},
        # #  "HuggingFaceTB/SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k-5e-5"
        # {"model": "SmolLM2-Instruct-16k-SeaLong-LongAlign-ST10k-v2-rope300k-5e-5", "training_length": 16384},
        # #  "/fsx/loubna/data/SmolLM2-1.7B-Instruct-rope500k"
        # {"model": "SmolLM2-1.7B-Instruct-rope500k", "training_length": 16384},

        # /fsx/loubna/data/SmolLM2-1.7B-Instruct-rope600k-32k
        {"model": "SmolLM2-1.7B-Instruct-rope600k-32k", "training_length": 32768},
        # /fsx/loubna/data/SmolLM2-1.7B-Instruct-rope500k-32k
        {"model": "SmolLM2-1.7B-Instruct-rope500k-32k", "training_length": 32768},
        # /fsx/loubna/data/SmolLM2-1.7B-Instruct-rope700k-32k
        {"model": "SmolLM2-1.7B-Instruct-rope700k-32k", "training_length": 32768},
        # /fsx/loubna/data/SmolLM2-1.7B-Instruct-rope800k-32k
        {"model": "SmolLM2-1.7B-Instruct-rope800k-32k", "training_length": 32768},

        # {"model": "SmolLM2-1.7B-final", "training_length": 2048, "use_chat_template": False},
        # {"model": "SmolLM2-1.7B-Intermediate-SFT-v2", "training_length": 2048},
        # {"model": "granite-3.0-1b-a400m-base", "training_length": 4096, "use_chat_template": False},
        # {"model": "granite-3.0-1b-a400m-instruct", "training_length": 4096},
        # {"model": "smollmv2-1.7B-32k-final", "training_length": 32768, "use_chat_template": False},
        # {"model": "smollmv2-1.7B-16k-final", "training_length": 16384, "use_chat_template": False},
        # {"model": "smollmv2-lc-base-16k-lr1e-5", "training_length": 16384, "use_chat_template": False},
        # {"model": "smollmv2-lc-base-16k-lr2e-5", "training_length": 16384, "use_chat_template": False},
        # {"model": "smollmv2-lc-base-8k-lr2e-5-800steps", "training_length": 8192, "use_chat_template": False},
        # {"model": "smollmv2-lc-base-8k-lr2e-5", "training_length": 8192, "use_chat_template": False},
        # {"model": "smollmv2-lc-co1-base-8k-16k-lr1e-5-400steps", "training_length": 16384, "use_chat_template": False},
        # {"model": "smollmv2-lc-co1-base-16k-32k-lr1e-5-400steps", "training_length": 32768, "use_chat_template": False},
        # {"model": "smollmv2-lc-co1-base-8k-32k-lr1e-5-400steps", "training_length": 32768, "use_chat_template": False},
        # {"model": "smollmv2-lc-m2-base-16k-lr1e-5-200steps", "training_length": 16384, "use_chat_template": False},
        # {"model": "smollmv2-lc-m2-base-16k-lr1e-5-400steps", "training_length": 16384, "use_chat_template": False},
        # {"model": "smollmv2-lc-m2-base-16k-lr1e-5-800steps", "training_length": 16384, "use_chat_template": False},
        # {"model": "smollmv2-lc-m2-base-8k-lr1e-5-800step", "training_length": 8192, "use_chat_template": False},
        # {"model": "smollmv2-lc-m2-base-8k-lr1e-5-400steps", "training_length": 8192, "use_chat_template": False},
        # {"model": "smollmv2-lc-m2-base-8k-lr1e-5-200steps", "training_length": 8192, "use_chat_template": False},
        #
        # {"model": "smollmv2-temp-1.7B-16k", "training_length": 16384, "use_chat_template": False},
        # {"model": "smollm2-lc-merge-no-base", "training_length": 16384, "use_chat_template": False},
        # {"model": "smollmv2-lc-m4-base-16k-lr1e-5-800steps", "training_length": 16384, "use_chat_template": False},
        #
        # {"model": "decay-75B_8k_lc", "training_length": 8192, "use_chat_template": False},
        # {"model": "decay-200B_8k_lc-5100000", "training_length": 8192, "use_chat_template": False},
        # {"model": "decay-200B_8k_lc-5112000", "training_length": 8192, "use_chat_template": False},
        # {"model": "decay-200B_8k_lc-5120000", "training_length": 8192, "use_chat_template": False},

        # {"model": "lc-50B-8k-to-16k-5221000", "training_length": 16384, "use_chat_template": False},
        # {"model": "lc-50B-8k-to-16k-5225000", "training_length": 16384, "use_chat_template": False},
        # {"model": "lc-50B-8k-to-16k-5229000", "training_length": 16384, "use_chat_template": False},
        # {"model": "lc-50B-8k-to-16k-5233000", "training_length": 16384, "use_chat_template": False},

        # "HuggingFaceTB/fix-rope-50B-8k-5216000" # 39
        #   "HuggingFaceTB/fix-rope-50B-8k-5218000" # 40
        #   "HuggingFaceTB/fix-rope-50B-8k-5220000" # 41
        # {"model": "fix-rope-50B-8k-5216000", "training_length": 8192, "use_chat_template": False},
        # {"model": "fix-rope-50B-8k-5218000", "training_length": 8192, "use_chat_template": False},
        # {"model": "fix-rope-50B-8k-5220000", "training_length": 8192, "use_chat_template": False},
        #
        # # {"model": "ft-16k-1e5", "training_length": 16384, "use_chat_template": False},
        # {"model": "50B-16k-320k-rope", "training_length": 16384, "use_chat_template": False},
        # {"model": "big-50B-8k-130k-rope", "training_length": 8192, "use_chat_template": False},
        #
        # {"model": "135M-lc-6B", "training_length": 8192, "use_chat_template": False},
        # {"model": "360M-50B-8k-2592k", "training_length": 8192, "use_chat_template": False},
        # {"model": "50B-decay-last-try", "training_length": 8192, "use_chat_template": False},
        # {"model": "360M-lc-100k-rope-12B", "training_length": 8192, "use_chat_template": False},
        # {"model": "135M-lc-12B", "training_length": 8192, "use_chat_template": False},
        # {"model": "135M-lc-100k-rope-12B", "training_length": 8192, "use_chat_template": False},
        #
        # {"model": "**SFT**", "training_length": 99999},
        #
        # {"model": "smollm2-1.7B-16k-m4-magpieu-ifeval", "training_length": 16384},
        # {"model": "smollm2-1.7B-16k-m4-magpieu-ifeval-longalign", "training_length": 16384},
        # {"model": "smollm2-1.7B-16k-m4-mix5-ep1", "training_length": 16384},
        # {"model": "smollm2-1.7B-16k-m4-mix5-longalign-ep1", "training_length": 16384},
        #
        # {"model": "smollm2-1.7B-decay-75B-8k-magpieu-ifeval-longalign", "training_length": 8192},
        # {"model": "smollm2-1.7B-decay-75B-8k-mix5-longalign-ep1", "training_length": 8192},
        #
        # # "HuggingFaceTB/smollm2-8k-dpo-mix5-longalign-ultraf-ep1" # 36
        # #   "HuggingFaceTB/smollm2-8k-dpo-mix5-longalign-ultraf-ep2" # 37
        # #   "HuggingFaceTB/smollm2-18k-dpo-magpieu-ifeval-longalign-ultraf-ep3" # 38
        # {"model": "smollm2-18k-dpo-magpieu-ifeval-longalign-ultraf-ep3", "training_length": 8192},
        # {"model": "smollm2-8k-dpo-mix5-longalign-ultraf-ep1", "training_length": 8192},
        # {"model": "smollm2-8k-dpo-mix5-longalign-ultraf-ep2", "training_length": 8192},
        # {"model": "smollm2-1.7B-decay-75B-8k-mix5-longalign-rewritev2", "training_length": 8192},
        #
        # {"model": "smollm2-1.7B-8k-mix7-half-ep2-v2-dpo-ultraf-ep3", "training_length": 8192},
        #
        # {"model": "smollm2-1.7B-8k-mix7-ep2-v2-dpo-ultraf-ep3", "training_length": 8192},
    ]
    #MAX_LENGTH = 8192
    #MAX_LENGTH = 16384
    MAX_LENGTH = 32768

    # set your configs here
    # configs = ["configs/recall_short.yaml", "configs/rag_short.yaml", "configs/rerank_short.yaml",
    #            "configs/cite_short.yaml", "configs/longqa_short.yaml", "configs/summ_short.yaml",
    #            "configs/icl_short.yaml"]
    configs = ["recall.yaml", "icl.yaml", "longqa.yaml", "rag.yaml", "rerank.yaml"]
    if MAX_LENGTH == 8192:
        configs = [f"configs/smollm_8k/{c}" for c in configs]
    elif MAX_LENGTH == 16384:
        configs = [f"configs/smollm_16k/{c}" for c in configs]
    elif MAX_LENGTH == 32768:
        configs = [f"configs/smollm_32k/{c}" for c in configs]
    else:
        raise ValueError(f"No configs for max length {MAX_LENGTH}")
    datasets_configs = []
    for config in configs:
        c = yaml.safe_load(open(config))
        print(c)
        if isinstance(c["generation_max_length"], int):
            c["generation_max_length"] = ",".join([str(c["generation_max_length"])] * len(c["datasets"].split(",")))
        if isinstance(c["input_max_length"], int):
            c["input_max_length"] = ",".join([str(c["input_max_length"])] * len(c["datasets"].split(",")))
        for d, t, l, g in zip(c['datasets'].split(','), c['test_files'].split(','), c['input_max_length'].split(','),
                              c['generation_max_length'].split(',')):
            datasets_configs.append(
                {"dataset": d, "test_name": os.path.basename(os.path.splitext(t)[0]), "input_max_length": int(l),
                 "generation_max_length": int(g), "use_chat_template": c["use_chat_template"],
                 "max_test_samples": c["max_test_samples"], 'shots': c['shots']})

    df = []
    for model in tqdm(models_configs):
        if model["training_length"] < MAX_LENGTH:
            continue
        args = arguments()
        args.tag = "v1"  # SET YOUR TAG HERE
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
    print(all_df)
    lf_df = all_df.pivot_table(index=["model", "input_max_length", ], columns="dataset_simple", values="metric",
                               sort=False)
    lf_df = lf_df.reset_index()
    lf_df = lf_df[lf_df['input_max_length'] == MAX_LENGTH]

    custom_avgs = {
        "Recall": ["json_kv substring_exact_match", "ruler_niah_mk_2 ruler_recall", "ruler_niah_mk_3 ruler_recall",
                   "ruler_niah_mv ruler_recall"],
        "ICL": ['trec_coarse exact_match', 'trec_fine exact_match', 'banking77 exact_match', 'clinic150 exact_match',
                'nlu exact_match'],
        # "Cite": ['alce_asqa str_em', 'alce_asqa citation_rec', 'alce_asqa citation_prec', 'alce_qampari qampari_rec_top5', 'alce_qampari citation_rec', 'alce_qampari citation_prec', ],
        "LongQA": ['infbench_qa rougeL_f1', 'infbench_choice exact_match', ],
        "RAG": ['nq substring_exact_match', 'hotpotqa substring_exact_match', 'popqa substring_exact_match',
                'triviaqa substring_exact_match', ],
        "Re-rank": ['msmarco_rerank_psg NDCG@10', ],
        # "Summ": ['infbench_sum gpt4-f1', 'multi_lexsum gpt4-f1', ],
        "Average-Real": ['RAG', 'ICL', 'Re-rank', 'LongQA'],
        "Average-All": ['Recall', 'RAG', 'ICL', 'Re-rank', 'LongQA'],
    }
    lf_df['Recall'] = np.mean(lf_df[custom_avgs['Recall']], axis=1)
    lf_df['RAG'] = np.mean(lf_df[custom_avgs['RAG']], axis=1)
    lf_df['Re-rank'] = np.mean(lf_df[custom_avgs['Re-rank']], axis=1)
    lf_df['LongQA'] = np.mean(lf_df[custom_avgs['LongQA']], axis=1)
    lf_df['ICL'] = np.mean(lf_df[custom_avgs['ICL']], axis=1)
    # # lf_df['Cite'] = np.mean(lf_df[custom_avgs['Cite']], axis=1)
    # # lf_df['Summ'] = np.mean(lf_df[custom_avgs['Summ']], axis=1)
    lf_df['Average-Real'] = np.mean(lf_df[custom_avgs['Average-Real']], axis=1)
    lf_df['Average-All'] = np.mean(lf_df[custom_avgs['Average-All']], axis=1)

    cols = ["model", "input_max_length", "Average-Real", "Average-All", "Recall", "ICL", "LongQA", "RAG", "Re-rank"]
    for c in lf_df.columns:
        if c not in cols:
            cols.append(c)
    lf_df = lf_df[cols]

    print(lf_df.to_csv(index=False))
    # import pdb; pdb.set_trace()

