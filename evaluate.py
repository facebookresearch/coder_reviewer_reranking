# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import tempfile
from datasets import load_metric
from tqdm import tqdm
import pickle as pkl
from data import MBPPGoogleDataset
from execution import Command
import sys
from utils import time_limit


""" dataset keys: src, trg_prediction, reference """


def evaluate_charbleu(dataset):
    bleu = load_metric("bleu")
    predictions = [[ch for ch in item["trg_prediction"]] for item in dataset]
    references = [[[ch for ch in item["reference"]]] for item in dataset]
    return bleu.compute(predictions=predictions, references=references)


""" dataset keys: src, trg_prediction, reference (only trg_prediction useful) """


def evaluate_spider_with_cached_results(selected):
    all_pred_results = [item["execution_result"] for item in selected]
    all_gold_results = pkl.load(
        open(
            "./dataset/spider/cached_gold_results.pkl",
            "rb",
        )
    )

    total_correct = 0
    for p_res, g_res in tqdm(
        zip(all_pred_results, all_gold_results),
        total=len(all_gold_results),
    ):
        total_correct += int(p_res[1] == g_res)

    return total_correct / len(all_gold_results)


def evaluate_one_mbpp(args, tempdir, dataset, timeout):
    i, item = args
    if "execution_result_full_pass" in dataset[i]:
        return int(
            all(
                isinstance(x[1], bool) and x[1] == True
                for x in dataset[i]["execution_result_full_pass"]
            )
        )
    else:
        test_cases = item["test_list"]
        test_setups = item["test_setup_code"]
        code = dataset[i]["trg_prediction"]
        # write code to file
        with open(f"{tempdir.name}/code-{i}.py", "w") as fout:
            print(code, file=fout)
            print(test_setups, file=fout)
            for case in test_cases:
                print(case, file=fout)
            fout.close()
        command = Command(f"python {tempdir.name}/code-{i}.py >/dev/null 2>&1")
        execution_result = command.run(timeout=timeout) == 0
        return execution_result


from functools import partial
from multiprocessing import Pool

""" dataset keys: src, trg_prediction, reference (only trg_prediction useful) """


def evaluate_google_mbpp(
    dataset,
    reference_path,
    split="test",
    timeout=10,
    return_details=False,
    num_procs=1,
    verbose=False,
):
    references = MBPPGoogleDataset(reference_path)
    assert len(dataset) == len(references.raw_data[split])
    tempdir = tempfile.TemporaryDirectory()
    passed_information = list()
    partial_evalutate_one = partial(
        evaluate_one_mbpp, tempdir=tempdir, dataset=dataset, timeout=timeout
    )

    if num_procs > 1:
        with Pool(processes=num_procs) as pool:
            for result_json in tqdm(
                pool.imap(
                    partial_evalutate_one, list(enumerate(references.raw_data[split]))
                ),
                total=len(references.raw_data[split]),
                leave=False,
                disable=not verbose,
            ):
                passed_information.append(result_json)
    else:
        for args in tqdm(
            list(enumerate(references.raw_data[split])), disable=not verbose
        ):
            passed_information.append(partial_evalutate_one(args))
    tempdir.cleanup()
    if return_details:
        return passed_information
    else:
        return sum(passed_information) / len(passed_information)


def evaluate_humaneval(dataset):
    all_passed = [d["execution_result_full_pass"] for d in dataset]
    return sum(all_passed) / len(all_passed)
