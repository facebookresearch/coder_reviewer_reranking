# Copyright (c) Meta Platforms, Inc. and affiliates.

import bashlex
import collections
import json
import pickle
import numpy as np
import os
import random
from glob import glob
from nltk.translate.bleu_score import sentence_bleu
from evaluate import (
    evaluate_charbleu,
    evaluate_google_mbpp,
    evaluate_spider_with_cached_results,
    evaluate_humaneval,
)
from tqdm import tqdm, trange
from pyminifier_canonicalize import clean_comment
import torch
from argparse import ArgumentParser
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from datasets import load_metric


class MultiSampleSelector(object):
    def __init__(
        self,
        paths,
        split="dev",
        tag="",
        model="",
        dataset="",
        verbose=False,
        no_rejection=False,
    ):
        self.paths = (
            list(sorted(glob(paths, recursive=True)))
            if isinstance(paths, str)
            else list(sorted(paths))
        )
        self.split = split
        self.data = collections.defaultdict(list)
        self.args = collections.defaultdict(list)
        self.tag = tag
        self.model = model
        self.dataset = dataset
        self.verbose = verbose
        for i, path in tqdm(
            enumerate(self.paths),
            total=len(self.paths),
            desc="loading jsons",
            disable=not self.verbose,
        ):
            self.args[i] = pickle.load(open(f"{self.paths[0]}/configs.pkl", "rb"))
            idx = 0
            if self.tag != "":
                file_path = f"{path}/{split}-{idx}-{tag}.jsonl"
            else:
                file_path = f"{path}/{split}-{idx}.jsonl"
            while os.path.exists(file_path):
                self.data[i, idx].extend([json.loads(x) for x in open(file_path)])
                idx += 1
                if self.tag != "":
                    file_path = f"{path}/{split}-{idx}-{tag}.jsonl"
                else:
                    file_path = f"{path}/{split}-{idx}.jsonl"

        print(f"{len(self.data)} cached samples")

        for path_id, sample_id in tqdm(
            self.data, desc="loading logprobs", disable=not self.verbose
        ):

            if (
                self.paths[path_id].find("nl2bash") != -1
            ):  # NL2bash data, exec simulation

                if self.tag != "":
                    file_name = f"{path}/{split}-{sample_id}-{tag}"
                else:
                    file_name = f"{path}/{split}-{sample_id}"
                exec_results = pickle.load(open(f"{file_name}.exec.pkl", "rb"))
                simulate_exec_results = pickle.load(
                    open(f"{file_name}.exec.simulate.pkl", "rb")
                )
                splitted_exec_results = pickle.load(
                    open(f"{file_name}.exec.splitted.pkl", "rb")
                )
                char_bleus = pickle.load(open(f"{file_name}.exec.bleu.pkl", "rb"))

            for item_i, item in enumerate(self.data[path_id, sample_id]):
                if no_rejection:
                    item["not_degenerate"] = True
                else:
                    # implementing degenerate solution rejection
                    if self.dataset in ["codet_humaneval", "mbpp_sanitized"]:
                        item["not_degenerate"] = filter_empty(
                            item, remove_function_header=False
                        ) and filter_repeat(item["trg_prediction"])
                    elif self.dataset in ["mbpp"]:
                        item["not_degenerate"] = filter_empty(
                            item, remove_function_header=True
                        )
                    elif self.dataset in ["spider", "nl2bash"]:
                        item["not_degenerate"] = len(
                            item["trg_prediction"]
                        ) != 0 and filter_repeat(item["trg_prediction"])
                    else:
                        raise ValueError("Invalid Dataset.")
                avg_logprob, sum_logprob = self.extract_logprob_stats(item, path_id)
                item["avg_logprob"] = avg_logprob
                item["sum_logprob"] = sum_logprob

                reverse_logprob = self.extract_reverse_logprob(item, path_id)
                (
                    item["sum_reverse_logprob"],
                    item["avg_reverse_logprob"],
                ) = reverse_logprob

                if (
                    self.paths[path_id].find("nl2bash") != -1
                ):  # NL2bash data, exec simulation
                    item["executable"] = exec_results[item_i]
                    item["trg_prediction_splitted"] = splitted_exec_results[item_i]
                    item["execution_result_simulated"] = simulate_exec_results[item_i]
                    item["charbleu"] = char_bleus[item_i]

    def extract_reverse_logprob(self, item, path_id):
        if "prompt_reverse_logprobs" not in item:
            return 0, 0
        logprobs = item["prompt_reverse_logprobs"]
        return np.sum(logprobs), np.mean(logprobs)

    def extract_logprob_stats(self, item, path_id):
        current_seq = ""
        if "codex" in self.model:
            extracted_position = None
            for i, _ in enumerate(item["tokens"]):
                current_seq += item["tokens"][i]
                end_template = self.args[path_id].end_template
                if isinstance(end_template, list):
                    end_template = ""
                if (
                    current_seq.find(item["trg_prediction"]) != -1
                    and current_seq.find(end_template) != -1
                ):
                    extracted_position = i + 1
                    break
            logprobs = (
                item["logprobs"][:extracted_position]
                if extracted_position is not None
                else item["logprobs"]
            )
            logprobs = list(
                filter(lambda x: x < 0, logprobs)
            )  # handle potential codex bug on positive log probability
        else:
            logprobs = item["logprobs"]
        return np.mean(logprobs), np.sum(logprobs)

    def select(
        self, ids=None, key_extractor=lambda x: x["avg_logprob"], return_keys=False
    ):
        if ids is None:
            ids = self.data.keys()
        ids = list(sorted(ids))
        n_examples = len(self.data[ids[0]])
        selected_examples = list()
        sample_keys = collections.defaultdict(list)
        for i in range(n_examples):
            max_key = None
            selected_item = None
            for idx in ids:
                item = self.data[idx][i]
                key = key_extractor(item)
                sample_keys[idx].append(key)
                if max_key is None or key > max_key:
                    max_key = key
                    selected_item = item
            assert selected_item is not None
            selected_examples.append(selected_item)
        if return_keys:
            return selected_examples, sample_keys
        else:
            return selected_examples


class ExecutionBasedMultiSampleSelector(MultiSampleSelector):
    def __init__(
        self,
        paths,
        split="dev",
        execution_type=None,
        tag="",
        model="",
        verbose=False,
        dataset="",
        no_rejection=False,
    ):
        super().__init__(
            paths,
            split=split,
            tag=tag,
            model=model,
            verbose=verbose,
            dataset=dataset,
            no_rejection=no_rejection,
        )
        self.execution_type = execution_type
        load_execution(self.data, self.paths, split, self.tag)


class IntraMultiSampleSelector(MultiSampleSelector):
    def __init__(
        self,
        paths,
        split="dev",
        tag="",
        model="",
        verbose=False,
        dataset="",
        no_rejection=False,
    ):
        super().__init__(
            paths,
            split=split,
            tag=tag,
            model=model,
            verbose=verbose,
            dataset=dataset,
            no_rejection=no_rejection,
        )

    def select(
        self,
        ids=None,
        key_extractor=None,
        second_key_extractor=None,
        return_keys=False,
        quantile_threshold=None,
    ):
        if ids is None:
            ids = self.data.keys()
        elif isinstance(ids, int):
            ids = [
                (i, j) for i in set(x[0] for x in self.data.keys()) for j in range(ids)
            ]
        ids = list(sorted(ids))
        id_set = set(ids)
        sample_keys = collections.defaultdict(list)
        # print(f'Selecting Samples from IDs: {ids}')
        n_examples = len(self.data[ids[0]])
        selected_examples = list()
        for i in range(n_examples):
            max_key = None
            selected_item = None
            if quantile_threshold is not None:
                filtered_ids = []
                all_second_key = []
                for idx in ids:
                    selected_item = None
                    item = self.data[idx][i]
                    all_second_key.append(
                        second_key_extractor(item)
                        if second_key_extractor is not None
                        else 0
                    )
                threshold = np.quantile(all_second_key, quantile_threshold)
                for idx_i, idx in enumerate(ids):
                    if all_second_key[idx_i] >= threshold:
                        filtered_ids.append(idx)
            else:
                filtered_ids = ids
            for idx in filtered_ids:
                item = self.data[idx][i]
                first_keys = list()
                for grndtruth_idx in filtered_ids:
                    grndtruth_item = self.data[grndtruth_idx][i]
                    key = key_extractor(item, grndtruth_item)
                    first_keys.append(key)
                first_key = sum(first_keys)
                second_key = (
                    second_key_extractor(item)
                    if second_key_extractor is not None
                    else 0
                )
                current_key = (first_key, second_key)
                item["mbr_key"] = current_key
                sample_keys[idx].append(current_key)
                if max_key is None or current_key > max_key:
                    max_key = current_key
                    selected_item = item
            assert selected_item is not None
            selected_examples.append(selected_item)
        if return_keys:
            return selected_examples, sample_keys
        else:
            return selected_examples


class ExecutionBasedIntraMultiSampleSelector(IntraMultiSampleSelector):
    def __init__(
        self,
        paths,
        split="dev",
        execution_type=None,
        tag="",
        model="",
        verbose=False,
        dataset="",
        no_rejection=False,
    ):
        super().__init__(
            paths,
            split=split,
            tag=tag,
            model=model,
            verbose=verbose,
            dataset=dataset,
            no_rejection=no_rejection,
        )
        self.execution_type = execution_type
        load_execution(self.data, self.paths, split, self.tag)


def filter_empty(x, remove_function_header=False):
    code = x["trg_prediction"]
    if remove_function_header:
        code = "\n".join(
            [l for l in code.split("\n") if not l.strip().startswith("def")]
        )
    try:
        code = clean_comment(code)
    except:
        code = ""
    return code.strip() not in ["", "pass", "return"]


def filter_repeat(x, threshold=0.25):
    import zlib

    bytes_x = bytes(x, encoding="utf-8")
    comp_x = zlib.compress(bytes_x)
    return len(comp_x) / len(bytes_x) > threshold


def load_execution(data_dict, paths, split, tag):
    if "spider" in paths[0]:
        exec_list = [
            ("exec.pkl", "execution_result"),
        ]
    else:
        exec_list = [
            ("exec.pkl", "execution_result"),
            ("execfull.pkl", "execution_result_full"),
            ("execfullpass.pkl", "execution_result_full_pass"),
            ("gen.execfull.pkl", "gen_execution_result_full"),
        ]
    for suffix, result_name in exec_list:
        for i, idx in data_dict:
            if tag == "":
                out_name = f"{split}-{idx}.{suffix}"
            else:
                out_name = f"{split}-{idx}-{tag}.{suffix}"
            path = paths[i]
            if suffix != "gen.execfull.pkl" or os.path.exists(f"{path}/{out_name}"):
                execution_results = pickle.load(open(f"{path}/{out_name}", "rb"))
            assert len(execution_results) == len(data_dict[i, idx])
            for j, execution_result in enumerate(execution_results):
                data_dict[i, idx][j][result_name] = execution_result


"""equivalence checking functions"""
# base equavalence checking function
def single_exec_result_matching(exec_x, exec_y, good_execution_result):
    try:
        if (
            exec_x[0] == good_execution_result
            and exec_y[0] == good_execution_result
            and exec_x[1] == exec_y[1]
        ):
            return 1
        else:
            return 0
    except:
        return 0


def multi_exec_result_matching(exec_x, exec_y, good_execution_result):
    try:
        same_output_count = 0
        if exec_x[0] == good_execution_result and exec_y[0] == good_execution_result:
            for ex, ey in exec_x[0], exec_y[1]:
                same_output_count += 1
            return same_output_count
        else:
            return 0
    except:
        return 0


# first assertion call matching
def execution_selection_function(x, y, good_execution_result=0):
    exec_x, exec_y = x["execution_result"], y["execution_result"]
    return single_exec_result_matching(exec_x, exec_y, good_execution_result)


def multi_execution_selection_function(x, y, good_execution_result=0):
    exec_x, exec_y = x["gen_execution_result_full"], y["gen_execution_result_full"]
    return sum(
        [
            single_exec_result_matching(single_x, single_y, good_execution_result)
            for single_x, single_y in zip(exec_x, exec_y)
        ]
    )


# just executability checking
def executability_selection_function(x, good_execution_result=0):
    exec_res = x["execution_result"]
    return exec_res[0] == good_execution_result


def multi_executability_selection_function(x, good_execution_result=0):
    exec_res = x["gen_execution_result_full"]
    return sum([e[0] == good_execution_result for e in exec_res])


def bleu_selection_function(x, y):
    return sentence_bleu(
        [[ch for ch in x["trg_prediction"]]], [ch for ch in y["trg_prediction"]]
    )


def token_bleu_selection_function(x, y):
    return sentence_bleu([x["trg_prediction"].split()], y["trg_prediction"].split())


def bash_execution_tokenbleu_selection_function(x, y):
    if not x["executable"] or not y["executable"]:
        return 0
    x = x["trg_prediction_splitted"]
    y = y["trg_prediction_splitted"]
    return sentence_bleu([x], y)


def get_mbpp_selector(
    criterion,
    mbpp_good_execution_result,
    use_multi_assertions=False,
    remove_function_header=False,
):
    if "$" in criterion:
        criterion = criterion.split("$")[0]
    secondary_key_function = None
    if not use_multi_assertions:
        exec_func = execution_selection_function
    else:
        exec_func = multi_execution_selection_function

    mbr_function = lambda x, y: all(
        [
            exec_func(x, y, mbpp_good_execution_result),
            x["not_degenerate"],
            y["not_degenerate"],
        ]
    )

    if not use_multi_assertions:
        executability_func = executability_selection_function
    else:
        executability_func = multi_executability_selection_function

    if criterion == "oracle":

        def get_oracle(x):
            if isinstance(x["execution_result_full_pass"], bool):
                return int(x["execution_result_full_pass"])
            elif isinstance(x["execution_result_full_pass"], list):
                return int(
                    all(
                        isinstance(exec_result[1], bool) and exec_result[1] == True
                        for exec_result in x["execution_result_full_pass"]
                    )
                )

        sample_selection_function = get_oracle
    elif criterion == "mbr_exec":
        sample_selection_function = mbr_function
        secondary_key_function = lambda x: x["sum_logprob"]
    elif criterion in [
        "sum_logprob",
        "avg_logprob",
        "avg_reverse_logprob",
        "sum_reverse_logprob",
    ]:
        criterion = criterion.split("-")[-1]
        sample_selection_function = lambda x: (
            x["not_degenerate"],
            x[criterion],
        )
    elif criterion == "random":
        sample_selection_function = lambda x: (
            x["not_degenerate"],
            random.random(),
        )
    elif criterion.startswith("avgreverselogprob-ensemble#"):
        alpha = float(criterion.split("#")[1])
        sample_selection_function = lambda x: (
            x["not_degenerate"],
            x["avg_reverse_logprob"] * alpha + x["avg_logprob"] * (1 - alpha),
        )
    elif criterion.startswith("sumreverselogprob-ensemble#"):
        alpha = float(criterion.split("#")[1])
        sample_selection_function = lambda x: (
            x["not_degenerate"],
            x["sum_reverse_logprob"] * alpha + x["sum_logprob"] * (1 - alpha),
        )
    elif criterion in [
        "executability-sum_logprob",
        "executability-avg_logprob",
        "executability-avg_reverse_logprob",
        "executability-sum_reverse_logprob",
    ]:
        criterion = criterion.split("-")[-1]
        sample_selection_function = lambda x: (
            x["not_degenerate"],
            executability_func(x, mbpp_good_execution_result),
            x[criterion],
        )
    elif criterion == "executability-random":
        sample_selection_function = lambda x: (
            x["not_degenerate"],
            executability_func(x, mbpp_good_execution_result),
            random.random(),
        )
    elif criterion.startswith("executability-avgreverselogprob-ensemble#"):
        alpha = float(criterion.split("#")[1])
        sample_selection_function = lambda x: (
            executability_func(x, mbpp_good_execution_result),
            x["not_degenerate"],
            x["avg_reverse_logprob"] * alpha + x["avg_logprob"] * (1 - alpha),
        )
    elif criterion.startswith("executability-sumreverselogprob-ensemble#"):
        alpha = float(criterion.split("#")[1])
        sample_selection_function = lambda x: (
            executability_func(x, mbpp_good_execution_result),
            x["not_degenerate"],
            x["sum_reverse_logprob"] * alpha + x["sum_logprob"] * (1 - alpha),
        )
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return sample_selection_function, secondary_key_function


"""
    select and evaluate a group in batch
    required keys:
        data_split: 'train', 'dev' or 'test'
        temperature: 0.1 .. 1.0
        criterion: 'mbr_exec' ... see full options in the function
        data_path: root data path for the task
        n_samples: number of candidates
        rand_seed: random seed for one experiment
"""


def bootstrap_mbpp(args):
    mbpp_good_execution_result = 0
    data_path = f"{args.data_path}/seed-*/**/*-{args.temperature}/"

    multisample_selector = ExecutionBasedMultiSampleSelector(
        data_path,
        args.data_split,
        "mbpp",
        tag=args.tag,
        model=args.model,
        dataset="mbpp",
        verbose=args.verbose,
        no_rejection=args.no_rejection,
    )
    intrasample_selector = ExecutionBasedIntraMultiSampleSelector(
        data_path,
        args.data_split,
        "mbpp",
        tag=args.tag,
        model=args.model,
        dataset="mbpp",
        verbose=args.verbose,
        no_rejection=args.no_rejection,
    )

    id_keys = list(multisample_selector.data.keys())

    acc_dict = collections.defaultdict(list)
    std_dict = collections.defaultdict(list)
    for crit in args.criteria:
        sample_selection_function, secondary_key_function = get_mbpp_selector(
            crit, mbpp_good_execution_result, remove_function_header=True
        )
        if "mbr" in crit:
            selector = intrasample_selector
        else:
            selector = multisample_selector

        step_results = collections.defaultdict(list)
        for sub_n_samples in trange(
            args.num_samples_start, args.num_samples_end, args.num_samples_gap
        ):
            random.seed(args.seed)
            np.random.seed(args.seed)
            for bootstrap_i in range(args.num_bootstraps):
                # id_is = np.random.choice(list(range(len(id_keys))), size=sub_n_samples, replace=True)
                # ids = [id_keys[i] for i in id_is]
                ids = random.sample(id_keys, sub_n_samples)
                if secondary_key_function is not None:
                    if "$" in crit:
                        quantile_threshold = float(crit.split("$")[1])
                    else:
                        quantile_threshold = None
                    selected = selector.select(
                        ids,
                        sample_selection_function,
                        secondary_key_function,
                        quantile_threshold=quantile_threshold,
                    )
                else:
                    selected = selector.select(ids, sample_selection_function)
                result = evaluate_google_mbpp(
                    selected, "dataset/mbpp/mbpp.jsonl", "test", verbose=args.verbose
                )
                step_results[sub_n_samples].append(result)
        for k, v in step_results.items():
            acc_dict[crit].append(np.mean(v))
            std_dict[crit].append(np.std(v))
    return (acc_dict, std_dict)


def get_nl2bash_selector(criterion):
    import random

    secondary_key_function = None
    if criterion == "oracle":
        sample_selection_function = lambda x: x["charbleu"]
    elif criterion == "mbr_bleu":
        raise NotImplementedError
        # sample_selection_function = lambda x, y: bleu_selection_function(x, y, tag=tag, model=model)
    elif criterion == "mbr_tokenbleu":
        sample_selection_function = lambda x, y: all(
            [
                token_bleu_selection_function(x, y),
                x["not_degenerate"],
                y["not_degenerate"],
            ]
        )
    elif criterion == "mbr_exec_tokenbleu":
        sample_selection_function = lambda x, y: all(
            [
                bash_execution_tokenbleu_selection_function(x, y),
                x["not_degenerate"],
                y["not_degenerate"],
            ]
        )
        secondary_key_function = lambda x: x["sum_logprob"]
    elif criterion in [
        "sum_logprob",
        "avg_logprob",
        "avg_reverse_logprob",
        "sum_reverse_logprob",
    ]:
        criterion = criterion.split("-")[-1]
        sample_selection_function = lambda x: (
            x["not_degenerate"],
            x[criterion],
        )
    elif criterion == "random":
        sample_selection_function = lambda x: (
            x["not_degenerate"],
            random.random(),
        )
    elif criterion.startswith("avgreverselogprob-ensemble#"):
        alpha = float(criterion.split("#")[1])
        sample_selection_function = lambda x: (
            x["not_degenerate"],
            x["avg_reverse_logprob"] * alpha + x["avg_logprob"] * (1 - alpha),
        )
    elif criterion.startswith("sumreverselogprob-ensemble#"):
        alpha = float(criterion.split("#")[1])
        sample_selection_function = lambda x: (
            x["not_degenerate"],
            x["sum_reverse_logprob"] * alpha + x["sum_logprob"] * (1 - alpha),
        )
    elif criterion in [
        "executability-sum_logprob",
        "executability-avg_logprob",
        "executability-avg_reverse_logprob",
        "executability-sum_reverse_logprob",
    ]:
        criterion = criterion.split("-")[-1]
        sample_selection_function = lambda x: (
            x["not_degenerate"],
            x["executable"],
            x[criterion],
        )
    elif criterion == "executability-random":
        sample_selection_function = lambda x: (
            x["not_degenerate"],
            x["executable"],
            random.random(),
        )
    elif criterion.startswith("executability-avgreverselogprob-ensemble#"):
        alpha = float(criterion.split("#")[1])
        sample_selection_function = lambda x: (
            x["executable"],
            x["not_degenerate"],
            x["avg_reverse_logprob"] * alpha + x["avg_logprob"] * (1 - alpha),
        )
    elif criterion.startswith("executability-sumreverselogprob-ensemble#"):
        alpha = float(criterion.split("#")[1])
        sample_selection_function = lambda x: (
            x["executable"],
            x["not_degenerate"],
            x["sum_reverse_logprob"] * alpha + x["sum_logprob"] * (1 - alpha),
        )
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return sample_selection_function, secondary_key_function


def bootstrap_nl2bash(args):
    data_path = f"{args.data_path}/seed-*/**/*-{args.temperature}/"
    secondary_key_function = None

    intra_selector = IntraMultiSampleSelector(
        data_path,
        args.data_split,
        tag=args.tag,
        model=args.model,
        verbose=args.verbose,
        dataset="nl2bash",
        no_rejection=args.no_rejection,
    )
    multi_selector = MultiSampleSelector(
        data_path,
        args.data_split,
        tag=args.tag,
        model=args.model,
        verbose=args.verbose,
        dataset="nl2bash",
        no_rejection=args.no_rejection,
    )
    id_keys = list(intra_selector.data.keys())
    acc_dict = collections.defaultdict(list)
    std_dict = collections.defaultdict(list)

    for crit in args.criteria:
        sample_selection_function, secondary_key_function = get_nl2bash_selector(crit)
        if "mbr" in crit:
            selector = intra_selector
        else:
            selector = multi_selector

        step_results = collections.defaultdict(list)
        for sub_n_samples in trange(
            args.num_samples_start, args.num_samples_end, args.num_samples_gap
        ):
            random.seed(args.seed)
            np.random.seed(args.seed)
            for bootstrap_i in range(args.num_bootstraps):
                ids = random.sample(id_keys, sub_n_samples)
                if secondary_key_function is not None:
                    selected = selector.select(
                        ids, sample_selection_function, secondary_key_function
                    )
                else:
                    selected = selector.select(ids, sample_selection_function)
                result = evaluate_charbleu(selected)["bleu"]
                step_results[sub_n_samples].append(result)
        for k, v in step_results.items():
            acc_dict[crit].append(np.mean(v))
            std_dict[crit].append(np.std(v))

    return acc_dict, std_dict


def bootstrap_human_eval(args):
    humaneval_good_execution_result = 0
    data_path = f"{args.data_path}/seed-*/0-shot/*-{args.temperature}"
    if args.top_p != 1.0:
        data_path += f"-p{args.top_p}"
    if args.max_tokens != 512:
        data_path += f"-max{args.max_tokens}"

    multisample_selector = ExecutionBasedMultiSampleSelector(
        data_path,
        args.data_split,
        "humaneval",
        tag=args.tag,
        model=args.model,
        verbose=args.verbose,
        dataset=args.dataset,
        no_rejection=args.no_rejection,
    )
    intrasample_selector = ExecutionBasedIntraMultiSampleSelector(
        data_path,
        args.data_split,
        "humaneval",
        tag=args.tag,
        model=args.model,
        verbose=args.verbose,
        dataset=args.dataset,
        no_rejection=args.no_rejection,
    )

    id_keys = list(multisample_selector.data.keys())

    acc_dict = collections.defaultdict(list)
    std_dict = collections.defaultdict(list)
    for crit in args.criteria:
        sample_selection_function, secondary_key_function = get_mbpp_selector(
            crit,
            humaneval_good_execution_result,
            use_multi_assertions=args.use_generated_assertions,
        )
        if "mbr" in crit:
            selector = intrasample_selector
        else:
            selector = multisample_selector

        step_results = collections.defaultdict(list)
        for sub_n_samples in trange(
            args.num_samples_start, args.num_samples_end, args.num_samples_gap
        ):
            random.seed(args.seed)
            np.random.seed(args.seed)
            for bootstrap_i in range(args.num_bootstraps):
                ids = random.sample(id_keys, sub_n_samples)
                # id_is = np.random.choice(list(range(len(id_keys))), size=sub_n_samples, replace=True)
                # ids = [id_keys[i] for i in id_is]
                if secondary_key_function is not None:
                    if "$" in crit:
                        quantile_threshold = float(crit.split("$")[1])
                    else:
                        quantile_threshold = None
                    selected = selector.select(
                        ids,
                        sample_selection_function,
                        secondary_key_function,
                        quantile_threshold=quantile_threshold,
                    )
                else:
                    selected = selector.select(ids, sample_selection_function)
                result = evaluate_humaneval(selected)
                step_results[sub_n_samples].append(result)
        for k, v in step_results.items():
            acc_dict[crit].append(np.mean(v))
            std_dict[crit].append(np.std(v))
    return (acc_dict, std_dict)


def get_spider_selector(criterion, spider_good_execution_result=True):
    import random

    secondary_key_function = None
    if criterion == "mbr_exec":
        sample_selection_function = lambda x, y: execution_selection_function(
            x, y, spider_good_execution_result
        )
        secondary_key_function = lambda x: x["sum_logprob"]
    elif criterion == "mbr_exec_reverse":
        sample_selection_function = lambda x, y: execution_selection_function(
            x, y, spider_good_execution_result
        )
        secondary_key_function = lambda x: x["avg_reverse_logprob"]
    elif criterion == "mbr_exec_avglogp":
        sample_selection_function = lambda x, y: execution_selection_function(
            x, y, spider_good_execution_result
        )
        secondary_key_function = lambda x: x["avg_logprob"]
    elif criterion == "random":
        sample_selection_function = lambda x: random.random()
    elif criterion in [
        "sum_logprob",
        "avg_logprob",
        "avg_reverse_logprob",
        "sum_reverse_logprob",
    ]:
        sample_selection_function = lambda x: x[criterion]
    elif criterion.startswith("avgreverselogprob-ensemble#"):
        alpha = float(criterion.split("#")[1])
        sample_selection_function = lambda x: x["avg_reverse_logprob"] * alpha + x[
            "avg_logprob"
        ] * (1 - alpha)
    elif criterion.startswith("sumreverselogprob-ensemble#"):
        alpha = float(criterion.split("#")[1])
        sample_selection_function = lambda x: x["sum_reverse_logprob"] * alpha + x[
            "sum_logprob"
        ] * (1 - alpha)
    elif criterion == "mbr_bleu":
        sample_selection_function = lambda x, y: bleu_selection_function(x, y)
    elif criterion == "mbr_tokenbleu":
        sample_selection_function = lambda x, y: token_bleu_selection_function(x, y)
    elif criterion in [
        "executability-sum_logprob",
        "executability-avg_logprob",
        "executability-avg_reverse_logprob",
        "executability-sum_reverse_logprob",
    ]:
        criterion = criterion.split("-")[1]
        sample_selection_function = lambda x: (
            executability_selection_function(x, spider_good_execution_result),
            x[criterion],
        )
    elif criterion == "executability-random":
        sample_selection_function = lambda x: (
            executability_selection_function(x, spider_good_execution_result),
            random.random(),
        )
    elif criterion == "executability-mbr_bleu":
        sample_selection_function = (
            lambda x, y: bleu_selection_function(x, y)
            * x["execution_result"][0]
            * y["execution_result"][0]
        )
    elif criterion == "executability-mbr_tokenbleu":
        sample_selection_function = (
            lambda x, y: token_bleu_selection_function(x, y)
            * x["execution_result"][0]
            * y["execution_result"][0]
        )
    elif criterion.startswith("executability-avgreverselogprob-ensemble#"):
        alpha = float(criterion.split("#")[1])
        sample_selection_function = lambda x: (
            executability_selection_function(x, spider_good_execution_result),
            x["avg_reverse_logprob"] * alpha + x["avg_logprob"] * (1 - alpha),
        )
    elif criterion.startswith("executability-sumreverselogprob-ensemble#"):
        alpha = float(criterion.split("#")[1])
        sample_selection_function = lambda x: (
            executability_selection_function(x, spider_good_execution_result),
            x["sum_reverse_logprob"] * alpha + x["sum_logprob"] * (1 - alpha),
        )
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return sample_selection_function, secondary_key_function


def bootstrap_spider(args):
    spider_good_execution_result = True
    data_path = f"{args.data_path}/seed-*/**/*-{args.temperature}/"

    intrasample_selector = ExecutionBasedIntraMultiSampleSelector(
        data_path,
        "dev",
        "spider",
        tag=args.tag,
        model=args.model,
        verbose=args.verbose,
        dataset="spider",
        no_rejection=args.no_rejection,
    )
    multisample_selector = ExecutionBasedMultiSampleSelector(
        data_path,
        "dev",
        "spider",
        tag=args.tag,
        model=args.model,
        verbose=args.verbose,
        dataset="spider",
        no_rejection=args.no_rejection,
    )  # pre-execution for faster evaluation

    id_keys = list(multisample_selector.data.keys())

    acc_dict = collections.defaultdict(list)
    std_dict = collections.defaultdict(list)

    # preloading spider data to reduce io
    from dataset.spider_official.evaluation import (
        evaluate,
        build_foreign_key_map_from_json,
    )

    kmaps = build_foreign_key_map_from_json(
        "/private/home/tianyizzz/projects/mbr-exec/dataset/spider/tables.json"
    )
    with open("dataset/spider/dev_gold.sql") as f:
        glist = [l.strip().split("\t") for l in f.readlines() if len(l.strip()) > 0]

    all_args = []
    flat_accs = []
    for crit in args.criteria:
        if "mbr" in crit:
            selector = intrasample_selector
        else:
            selector = multisample_selector
        for sub_n_samples in range(
            args.num_samples_start, args.num_samples_end, args.num_samples_gap
        ):
            random.seed(args.seed)
            np.random.seed(args.seed)
            for bootstrap_i in range(args.num_bootstraps):
                id_is = np.random.choice(
                    list(range(len(id_keys))), size=sub_n_samples, replace=True
                )
                ids = [id_keys[i] for i in id_is]
                all_args.append(
                    (
                        ids,
                        crit,
                        selector,
                        sub_n_samples * args.num_bootstraps + bootstrap_i,
                        kmaps,
                        glist,
                    )
                )

    if args.num_procs > 1:
        print(f"running with {args.num_procs} processes.")
        from multiprocessing.pool import ThreadPool as Pool

        with Pool(processes=args.num_procs) as pool:
            for acc in tqdm(
                pool.imap(evaluate_spider_one, all_args, chunksize=1),
                total=len(all_args),
                desc=f"{crit}",
            ):
                flat_accs.append(acc)
    else:
        for data in tqdm(all_args, total=len(all_args)):
            flat_accs.append(evaluate_spider_one(data))

    acc_idx = 0
    for crit in args.criteria:
        step_results = collections.defaultdict(list)
        for sub_n_samples in trange(
            args.num_samples_start, args.num_samples_end, args.num_samples_gap
        ):
            for bootstrap_i in range(args.num_bootstraps):
                step_results[sub_n_samples].append(flat_accs[acc_idx])
                acc_idx += 1
        for k, v in step_results.items():
            acc_dict[crit].append(np.mean(v))
            std_dict[crit].append(np.std(v))
    return (acc_dict, std_dict)


def evaluate_spider_one(args):
    ids, crit, selector, bootstrap_i, kmaps, glist = args
    sample_selection_function, secondary_key_function = get_spider_selector(crit)
    if secondary_key_function is not None:
        selected = selector.select(
            ids, sample_selection_function, secondary_key_function
        )
    else:
        selected = selector.select(ids, sample_selection_function)
    acc = evaluate_spider_with_cached_results(selected)
    return acc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="./result_db")
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "codegen-2B",
            "codegen-2B-half",
            "codegen-6B",
            "codegen-6B-half",
            "codegen-16B-half",
            "incoder-1B",
            "incoder-1B-half",
            "incoder-6B",
            "incoder-6B-half",
            "codex001",
            "codex002",
            "codex-cushman",
        ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mbpp",
        choices=[
            "mbpp",
            "spider",
            "nl2bash",
            "humaneval",
            "codet_humaneval",
            "mbpp_sanitized",
        ],
    )
    parser.add_argument("--num_samples_start", type=int, default=24)
    parser.add_argument("--num_samples_end", type=int, default=25)
    parser.add_argument("--num_samples_gap", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=1)
    parser.add_argument("--num_bootstraps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid_search_alpha", action="store_true", default=False)
    parser.add_argument("--ablate", action="store_true", default=False)
    parser.add_argument("--no-rejection", action="store_true", default=False)
    parser.add_argument(
        "--use_generated_assertions", action="store_true", default=False
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="samples",
    )
    args = parser.parse_args()

    if args.temperature > 0:
        base_crits = ["sum_logprob", "avg_logprob", "avg_reverse_logprob", "random"]
        if args.grid_search_alpha:
            for i in range(1, 10):
                base_crits.append(
                    f"sumreverselogprob-ensemble#0.{i}",
                )
            for i in range(1, 10):
                base_crits.append(
                    f"avgreverselogprob-ensemble#0.{i}",
                )
        else:
            base_crits.append(
                f"sumreverselogprob-ensemble#0.5",
            )
            base_crits.append(
                f"avgreverselogprob-ensemble#0.5",
            )
        all_crits = base_crits + ["executability-" + b for b in base_crits]
    else:
        all_crits = ["sum_logprob"]

    args.criteria = all_crits
    args.data_path = f"{args.data_path}/{args.model}/{args.dataset}"

    if args.dataset == "spider":
        args.criteria = all_crits + ["mbr_exec"]
        acc_dict, std_dict = bootstrap_spider(args)
    elif args.dataset == "mbpp":
        args.criteria = args.criteria + ["mbr_exec", "oracle"]
        args.data_split = "test"
        acc_dict, std_dict = bootstrap_mbpp(args)
    elif "humaneval" in args.dataset or args.dataset == "mbpp_sanitized":
        args.criteria = args.criteria + ["mbr_exec", "oracle"]
        args.data_split = "test"
        acc_dict, std_dict = bootstrap_human_eval(args)
    elif args.dataset == "nl2bash":
        args.criteria = args.criteria + ["mbr_exec_tokenbleu", "oracle"]
        args.data_split = "dev"
        acc_dict, std_dict = bootstrap_nl2bash(args)
    else:
        raise ValueError
    if args.tag != "":
        out_path = Path(
            f"{args.out_dir}/{args.dataset}-{args.model}-temp{args.temperature}-{args.tag}"
        )
    else:
        out_path = Path(
            f"{args.out_dir}/{args.dataset}-{args.model}-temp{args.temperature}"
        )
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(acc_dict, out_path / "acc.pt")
    torch.save(std_dict, out_path / "std.pt")
    print(f"saving to {out_path}")
    for crit in args.criteria:
        print(crit, f"{acc_dict[crit][-1]:.4f} {std_dict[crit][-1]:.2f}")
