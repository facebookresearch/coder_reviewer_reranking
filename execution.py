# Copyright (c) Meta Platforms, Inc. and affiliates.

import bashlex
import json
import os
import pickle
import regex
import signal
import subprocess
import tempfile
import threading
from datasets import load_metric
from glob import glob
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from dataset.human_eval.human_eval.evaluation import evaluate_functional_correctness
import numpy as np
from collections import Counter


from data import MBPPGoogleDataset, HumanEvalDataset, MBPPSanDataset
from utils_sql import *
from time import sleep


class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            self.process = subprocess.Popen(self.cmd, shell=True, preexec_fn=os.setsid)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            os.killpg(self.process.pid, signal.SIGTERM)
            thread.join()
        return self.process.returncode


class PythonFunctionExecutor(object):
    def __init__(self, function_content, function_call, timeout=10):
        self.function_content = function_content
        self.function_content = self.function_content.replace("</code>", "")
        self.function_call = function_call
        self.timeout = timeout

    def __call__(self, i, use_json=False):
        tempdir = tempfile.TemporaryDirectory()
        with open(f"{tempdir.name}/code-{i}.py", "w") as fout:
            print(self.function_content, file=fout)
            print(f"result = {self.function_call}", file=fout)
            print(f"import pickle", file=fout)
            print(
                f'pickle.dump(result, open("{tempdir.name}/execution_result-{i}.pkl", "wb"))',
                file=fout,
            )
        command = Command(f"python {tempdir.name}/code-{i}.py >/dev/null 2>&1")
        execution_status = command.run(timeout=self.timeout)
        if execution_status == 0:
            try:
                execution_results = pickle.load(
                    open(f"{tempdir.name}/execution_result-{i}.pkl", "rb")
                )
            except:
                execution_results = None
        else:
            execution_results = None
        tempdir.cleanup()
        return execution_status, execution_results


def mbpp_execute_one_assertion(args):
    data_item, code_item, i = args
    assertion = data_item[-1]
    command = regex.match(f"assert (.+)==.+", assertion).group(1)
    python_function = code_item["trg_prediction"]
    executor = PythonFunctionExecutor(python_function, command)
    execution_result = executor(i)
    return execution_result


def mbpp_execute_multiple_assertion(args):
    data_item, code_item, i = args
    execution_result = list()
    python_function = code_item["trg_prediction"]
    for assertion_i, assertion in enumerate(data_item[-1]):
        command = regex.match(f"assert (.+)==.+", assertion).group(1)
        executor = PythonFunctionExecutor(python_function, command)
        execution_result.append(executor(f"{i}-{assertion_i}"))
    return execution_result


def mbpp_execute_multiple_assertion_pass(args):
    data_item, code_item, i = args
    execution_result = list()
    python_function = code_item["trg_prediction"]
    for assertion_i, assertion in enumerate(data_item[-1]):
        command = regex.match(f"assert (.+==.+)", assertion).group(1)
        executor = PythonFunctionExecutor(python_function, f"({command})")
        execute_stats, execute_result = executor(f"{i}-{assertion_i}")
        # if isinstance(execute_result, tuple) and len(execute_result) == 2:
        #     execute_result = execute_result[0]
        # assert execute_result is None or isinstance(execute_result, bool)
        execution_result.append((execute_stats, execute_result))
    return execution_result


from multiprocessing import Pool


def execute_mbpp_google_folder(base_path, num_procs=10, verbose=False):
    # single assertion
    dataset = MBPPGoogleDataset(mode="assertion")
    for path in tqdm(
        glob(f"{base_path}/*jsonl"), leave=False, desc="exec one", disable=not verbose
    ):  # execute first assertion call
        if "with-reverse" in path:
            continue
        if os.path.exists(path.replace("jsonl", "exec.pkl")):
            continue
        split = os.path.basename(path).split("-")[0]
        execution_results = list()
        all_args = []
        for i, line in enumerate(open(path).readlines()):
            data_item = dataset.data[split][i]
            code_item = json.loads(line)
            all_args.append((data_item, code_item, i))
        if num_procs > 1:
            with Pool(processes=num_procs) as pool:
                for execution_result in pool.imap(mbpp_execute_one_assertion, all_args):
                    execution_results.append(execution_result)
        else:
            for execution_result in map(mbpp_execute_one_assertion, all_args):
                execution_results.append(execution_result)
        with open(path.replace("jsonl", "exec.pkl"), "wb") as fout:
            pickle.dump(execution_results, fout)
    # multiple assertions (cheating)
    dataset = MBPPGoogleDataset(mode="assertion-full")
    for path in tqdm(
        glob(f"{base_path}/*jsonl"),
        leave=False,
        desc="exec multiple",
        disable=not verbose,
    ):  # execute all assertion calls
        if "with-reverse" in path:
            continue
        if os.path.exists(path.replace("jsonl", "execfull.pkl")):
            continue
        split = os.path.basename(path).split("-")[0]
        execution_results = list()
        all_args = []
        for i, line in enumerate(open(path).readlines()):
            data_item = dataset.data[split][i]
            code_item = json.loads(line)
            import uuid

            all_args.append((data_item, code_item, str(uuid.uuid4())))
        if num_procs > 1:
            with Pool(processes=num_procs) as pool:
                for execution_result in pool.imap(
                    mbpp_execute_multiple_assertion, all_args
                ):
                    execution_results.append(execution_result)
        else:
            for execution_result in map(mbpp_execute_multiple_assertion, all_args):
                execution_results.append(execution_result)
        with open(path.replace("jsonl", "execfull.pkl"), "wb") as fout:
            pickle.dump(execution_results, fout)
    # multiple assertions (pass or fail)
    for path in tqdm(
        glob(f"{base_path}/*jsonl"),
        leave=False,
        desc="exec-multiple-pass",
        disable=not verbose,
    ):
        if "with-reverse" in path:
            continue
        if os.path.exists(path.replace("jsonl", "execfullpass.pkl")):
            continue
        split = os.path.basename(path).split("-")[0]
        execution_results = list()
        all_args = []
        for i, line in enumerate(open(path).readlines()):
            data_item = dataset.data[split][i]
            code_item = json.loads(line)
            all_args.append((data_item, code_item, i))
        if num_procs > 1:
            with Pool(processes=num_procs) as pool:
                for execution_result in pool.imap(
                    mbpp_execute_multiple_assertion_pass, all_args
                ):
                    execution_results.append(execution_result)
        else:
            for execution_result in map(mbpp_execute_multiple_assertion_pass, all_args):
                execution_results.append(execution_result)
        # with open(path.replace('jsonl', 'execfullpass.pkl'), 'rb') as fout:
        #     gt_execution_results = pickle.load(fout)
        # for i, (a, b) in enumerate(zip(execution_results, gt_execution_results)):
        #     if a != b:
        #         print(i, (a, b))
        with open(path.replace("jsonl", "execfullpass.pkl"), "wb") as fout:
            pickle.dump(execution_results, fout)


def execute_spider_folder(
    base_path,
    db_path="dataset/spider/database",
    gold_path="dataset/spider",
    table_path="dataset/spider/tables.json",
    timeout=10,
):
    kmaps = build_foreign_key_map_from_json(table_path)
    for path in glob(f"{base_path}/*jsonl"):
        if "with-reverse" in path:
            continue
        if os.path.exists(path.replace("jsonl", "exec.pkl")):
            continue
        execution_results = list()
        split = os.path.basename(path).split("-")[0]
        file_gold_path = f"{gold_path}/{split}_gold.sql"
        with open(file_gold_path) as f:
            glist = [l.strip().split("\t") for l in f if len(l.strip()) > 0]
        with open(path) as f:
            plist = [json.loads(l)["trg_prediction"] for l in f]
        for p_str, (_, db_name) in tqdm(list(zip(plist, glist))):
            db = os.path.join(db_path, db_name, db_name + ".sqlite")
            schema = Schema(get_schema(db))
            try:
                p_sql = get_sql(schema, p_str)
            except:
                # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
                p_sql = {
                    "except": None,
                    "from": {"conds": [], "table_units": []},
                    "groupBy": [],
                    "having": [],
                    "intersect": None,
                    "limit": None,
                    "orderBy": [],
                    "select": [False, []],
                    "union": None,
                    "where": [],
                }
            # rebuild sql for value evaluation
            kmap = kmaps[db_name]
            p_valid_col_units = build_valid_col_units(
                p_sql["from"]["table_units"], schema
            )
            p_sql = rebuild_sql_val(p_sql)
            p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
            execution_result = execute(db, p_str, p_sql, timeout)
            execution_results.append(execution_result)
        with open(path.replace("jsonl", "exec.pkl"), "wb") as fout:
            pickle.dump(execution_results, fout)


def simulate_bash_exec(command):
    return list(bashlex.split(command))


def execute_mbpp_google_folder_one(base_path, num_procs=5, verbose=False, tag=""):
    # single assertion
    path = str(base_path)
    dataset = MBPPGoogleDataset(mode="assertion")
    out_name = "exec.pkl"
    if not (os.path.exists(path.replace("jsonl", out_name))):
        split = os.path.basename(path).split("-")[0]
        execution_results = list()
        all_args = []
        for i, line in enumerate(open(path).readlines()):
            data_item = dataset.data[split][i]
            code_item = json.loads(line)
            all_args.append((data_item, code_item, i))
        if num_procs > 1:
            with Pool(processes=num_procs) as pool:
                for execution_result in tqdm(
                    pool.imap(mbpp_execute_one_assertion, all_args),
                    total=len(all_args),
                    leave=False,
                    disable=not verbose,
                    desc="exec on",
                ):
                    execution_results.append(execution_result)
        else:
            for execution_result in map(
                mbpp_execute_one_assertion, tqdm(all_args, disable=not verbose)
            ):
                execution_results.append(execution_result)
        with open(path.replace("jsonl", out_name), "wb") as fout:
            pickle.dump(execution_results, fout)
    # mltiple assertions (cheating)
    dataset = MBPPGoogleDataset(mode="assertion-full")
    path = str(base_path)
    out_name = "execfull.pkl"
    if not (os.path.exists(path.replace("jsonl", out_name))):
        split = os.path.basename(path).split("-")[0]
        execution_results = list()
        all_args = []
        for i, line in enumerate(open(path).readlines()):
            data_item = dataset.data[split][i]
            code_item = json.loads(line)
            all_args.append((data_item, code_item, i))
        if num_procs > 1:
            with Pool(processes=num_procs) as pool:
                for execution_result in tqdm(
                    pool.imap(mbpp_execute_multiple_assertion, all_args),
                    total=len(all_args),
                    leave=False,
                    disable=not verbose,
                    desc="exec all",
                ):
                    execution_results.append(execution_result)
        else:
            for execution_result in map(
                mbpp_execute_multiple_assertion, tqdm(all_args, disable=not verbose)
            ):
                execution_results.append(execution_result)
        with open(path.replace("jsonl", out_name), "wb") as fout:
            pickle.dump(execution_results, fout)
    # mltiple assertions (pass or fail)
    path = str(base_path)
    out_name = "execfullpass.pkl"
    if not (os.path.exists(path.replace("jsonl", out_name))):
        split = os.path.basename(path).split("-")[0]
        execution_results = list()
        all_args = []
        for i, line in enumerate(open(path).readlines()):
            data_item = dataset.data[split][i]
            code_item = json.loads(line)
            all_args.append((data_item, code_item, i))
        if num_procs > 1:
            with Pool(processes=num_procs) as pool:
                for execution_result in tqdm(
                    pool.imap(mbpp_execute_multiple_assertion_pass, all_args),
                    total=len(all_args),
                    leave=False,
                    disable=not verbose,
                    desc="pass or fail",
                ):
                    execution_results.append(execution_result)
        else:
            for execution_result in map(
                mbpp_execute_multiple_assertion_pass,
                tqdm(all_args, disable=not verbose),
            ):
                execution_results.append(execution_result)
        with open(path.replace("jsonl", out_name), "wb") as fout:
            pickle.dump(execution_results, fout)


def execute_spider_folder_one(
    base_path,
    db_path="dataset/spider/database",
    gold_path="dataset/spider",
    table_path="dataset/spider/tables.json",
    timeout=10,
    verbose=False,
    tag="",
):
    kmaps = build_foreign_key_map_from_json(table_path)
    path = str(base_path)
    out_name = "exec.pkl" if tag == "" else f"exec.pkl"
    if not (os.path.exists(path.replace("jsonl", f"{out_name}"))):
        execution_results = list()
        split = os.path.basename(path).split("-")[0]
        file_gold_path = f"{gold_path}/{split}_gold.sql"
        with open(file_gold_path) as f:
            glist = [l.strip().split("\t") for l in f if len(l.strip()) > 0]
        with open(path) as f:
            plist = [json.loads(l)["trg_prediction"] for l in f]

        count = 0
        for p_str, (_, db_name) in tqdm(
            list(zip(plist, glist)), disable=not verbose, desc="SQL exec"
        ):
            db = os.path.join(db_path, db_name, db_name + ".sqlite")
            schema = Schema(get_schema(db))
            try:
                p_sql = get_sql(schema, p_str)
            except:
                # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
                p_sql = {
                    "except": None,
                    "from": {"conds": [], "table_units": []},
                    "groupBy": [],
                    "having": [],
                    "intersect": None,
                    "limit": None,
                    "orderBy": [],
                    "select": [False, []],
                    "union": None,
                    "where": [],
                }
            # rebuild sql for value evaluation
            kmap = kmaps[db_name]
            p_valid_col_units = build_valid_col_units(
                p_sql["from"]["table_units"], schema
            )
            p_sql = rebuild_sql_val(p_sql)
            p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
            execution_result = execute(db, p_str, p_sql, timeout)
            execution_results.append(execution_result)
            count += 1
        with open(path.replace("jsonl", out_name), "wb") as fout:
            pickle.dump(execution_results, fout)


def humaneval_postprocess(
    completion,
):
    keep_lines = []
    for l in completion.split("\n"):
        if not l.startswith("print"):
            keep_lines.append(l)
    return "\n".join(keep_lines)


def humaneval_execute_one_assertion(problem):
    assertion = problem["assertion"]
    try:
        command = regex.match(f"assert (.+)==.+", assertion).group(1)
    except:
        command = regex.match(f"assert (.+)", assertion).group(1)
    python_function = problem["prompt"] + problem["completion"]
    executor = PythonFunctionExecutor(python_function, command)
    execution_result = executor(problem["task_id"].split("/")[1])
    return execution_result


def humaneval_execute_multiple_assertion(problem):
    execution_result = list()
    python_function = problem["prompt"] + problem["completion"]
    task_id = problem["task_id"].split("/")[1]
    for assertion_i, assertion in enumerate(problem["assertion"]):
        try:
            try:
                command = regex.match(f"assert (.+)==.+", assertion).group(1)
            except:
                command = regex.match(f"assert (.+)", assertion).group(1)
        except:
            print(problem["assertion"])
            print(problem["task_id"])
            breakpoint()
        executor = PythonFunctionExecutor(python_function, command)
        execution_result.append(executor(f"{task_id}-{assertion_i}"))
    return execution_result


def humaneval_execute_generated_assertion(problem):
    execution_result = list()
    python_function = problem["prompt"] + problem["completion"]
    task_id = problem["task_id"].split("/")[1]

    total_matched = 0
    for assertion_i, assertion in enumerate(problem["gen_assertion"]):
        matched = False
        for pattern in ["assert (.+)==.+", "assert (.+) is .+", "assert (.+)"]:
            try:
                command = regex.match(pattern, assertion).group(1)
                matched = True
                break
            except:
                pass

        if matched:
            executor = PythonFunctionExecutor(python_function, command)
            execution_result.append(executor(f"{task_id}-{assertion_i}"))
            total_matched += int(matched)

        if total_matched > 20:
            break
    return execution_result


def execute_humaneval_folder_one(
    base_path,
    timeout=10,
    verbose=False,
    tag="",
    num_procs=1,
    dataset_choice="humaneval",
):
    path = str(base_path)
    if dataset_choice in ["humaneval", "codet_humaneval"]:
        dataset_cls = HumanEvalDataset
        if dataset_choice == "codet_humaneval":
            dataset_problem_file = "dataset/human_eval/dataset/CodeTHumanEval.jsonl"
            assertion_file = "dataset/human_eval/dataset/HumanEval.jsonl"
        else:
            dataset_problem_file = "dataset/human_eval/dataset/HumanEval.jsonl"
            assertion_file = ""
    elif dataset_choice == "mbpp_sanitized":
        dataset_problem_file = "dataset/mbpp/mbpp_sanitized_for_code_generation.jsonl"
        assertion_file = ""
        dataset_cls = MBPPSanDataset
    else:
        raise ValueError("Invalid data choice")

    dataset = dataset_cls(
        path=dataset_problem_file, assertion_path=assertion_file, mode="assertion"
    )
    prompt_to_problem = {p["prompt"]: p for task_id, p in dataset.raw_data.items()}

    out_name = "exec.pkl"
    problem_with_completions = []
    for line in open(path).readlines():
        code_item = json.loads(line)
        problem = prompt_to_problem[code_item["prompt"]]
        problem["completion"] = humaneval_postprocess(code_item["trg_prediction"])
        problem_with_completions.append(problem)

    if not (os.path.exists(path.replace("jsonl", out_name))):
        execution_results = []
        if num_procs > 1:
            with Pool(processes=num_procs) as pool:
                for execution_result in pool.imap(
                    humaneval_execute_one_assertion, problem_with_completions
                ):
                    execution_results.append(execution_result)
        else:
            for execution_result in map(
                humaneval_execute_one_assertion, problem_with_completions
            ):
                execution_results.append(execution_result)
        with open(path.replace("jsonl", out_name), "wb") as fout:
            pickle.dump(execution_results, fout)

    dataset = dataset_cls(
        path=dataset_problem_file, assertion_path=assertion_file, mode="assertion-all"
    )
    prompt_to_problem = {p["prompt"]: p for task_id, p in dataset.raw_data.items()}
    problem_with_completions = []
    for line in open(path).readlines():
        code_item = json.loads(line)
        problem = prompt_to_problem[code_item["prompt"]]
        problem["completion"] = humaneval_postprocess(code_item["trg_prediction"])
        problem_with_completions.append(problem)

    out_name = "execfull.pkl"
    if not (os.path.exists(path.replace("jsonl", out_name))):
        execution_results = []
        if num_procs > 1:
            with Pool(processes=num_procs) as pool:
                for execution_result in pool.imap(
                    humaneval_execute_multiple_assertion, problem_with_completions
                ):
                    execution_results.append(execution_result)
        else:
            for execution_result in map(
                humaneval_execute_multiple_assertion, problem_with_completions
            ):
                execution_results.append(execution_result)
        with open(path.replace("jsonl", out_name), "wb") as fout:
            pickle.dump(execution_results, fout)

    out_name = "execfullpass.pkl"
    if not (os.path.exists(path.replace("jsonl", out_name))):
        results, pass_at_k, extras = evaluate_functional_correctness(
            samples=problem_with_completions,
            sample_file=None,
            k=[1],
            problem_file=dataset_problem_file,
            suppress=True,
            timeout=timeout,
        )
        all_passed = []
        for result in results.values():
            result.sort()
            passed = [r[1]["passed"] for r in result]
            assert len(passed) == 1
            all_passed.append(passed[0])
        with open(path.replace("jsonl", out_name), "wb") as fout:
            pickle.dump(all_passed, fout)
    else:
        all_passed = pickle.load(open(path.replace("jsonl", out_name), "rb"))


def execute_nl2bash_folder_one(
    base_path,
):
    bleu = load_metric("bleu")
    path = str(base_path)

    if all(
        (
            os.path.exists(path.replace(".jsonl", ".exec.pkl")),
            os.path.exists(path.replace(".jsonl", ".exec.splitted.pkl")),
            os.path.exists(path.replace(".jsonl", ".exec.simulate.pkl")),
            os.path.exists(path.replace(".jsonl", ".exec.bleu.pkl")),
        )
    ):
        # return
        pass

    all_exec_results = []
    all_exec_splitted_results = []
    all_simulate_exec = []
    all_char_bleu = []
    for line in tqdm(open(path).readlines()):
        code_item = json.loads(line)
        code_item["trg_prediction"]
        try:
            with time_limit(10):
                bashlex.parse(code_item["trg_prediction"])
                all_exec_results.append(True)
        except:
            all_exec_results.append(False)

        try:
            with time_limit(10):
                splitted_trg_pred = simulate_bash_exec(code_item["trg_prediction"])
        except:
            splitted_trg_pred = list()
        simulate_exec = Counter(splitted_trg_pred)
        all_exec_splitted_results.append(splitted_trg_pred)
        all_simulate_exec.append(simulate_exec)

        try:
            with time_limit(10):
                all_char_bleu.append(
                    bleu.compute(
                        predictions=[[ch for ch in code_item["reference"]]],
                        references=[[[ch for ch in code_item["trg_prediction"]]]],
                    )["bleu"]
                )
        except:
            all_char_bleu.append(0)

    with open(path.replace(".jsonl", ".exec.pkl"), "wb") as fout:
        pickle.dump(all_exec_results, fout)
    with open(path.replace(".jsonl", ".exec.splitted.pkl"), "wb") as fout:
        pickle.dump(all_exec_splitted_results, fout)
    with open(path.replace(".jsonl", ".exec.simulate.pkl"), "wb") as fout:
        pickle.dump(all_simulate_exec, fout)
    with open(path.replace(".jsonl", ".exec.bleu.pkl"), "wb") as fout:
        pickle.dump(all_char_bleu, fout)
