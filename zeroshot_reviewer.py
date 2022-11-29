# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path
import os
from glob import glob
from argparse import ArgumentParser
import html
import json
from utils import *
from tqdm import tqdm, trange
from data import HumanEvalDataset, rindex, extract_docstring
from functools import partial
from pyminifier_canonicalize import clean_comment, remove_print


def postprocess_func_only(code, tokens):
    lines = []
    for line in code.split("\n"):
        if len(line.strip()) > 0 and not line.startswith(" "):
            continue
        else:
            lines.append(line)

    code = "\n".join(lines)
    code = code.rstrip()

    curr = ""
    for i, tok in enumerate(tokens):
        curr += tok
        if len(curr) >= len(code):
            break

    return code, tokens[: i + 1]


def make_new_context(
    codex_data,
    problem,
    canonicalize=False,
    clean_print=False,
):
    prompt = codex_data["prompt"]
    code_sample = codex_data["trg_prediction"]
    if canonicalize:
        try:
            code_sample = clean_comment(code_sample)
        except:
            # static error
            code_sample = code_sample
    if clean_print:
        code_sample = remove_print(code_sample)
    func_name = problem["entry_point"]
    docstring, func_header, func_context, doc_start = extract_docstring(prompt)
    if canonicalize:
        func_header = func_header.replace(f"{func_name}(", "f(")
        docstring = docstring.replace(f"{func_name}(", "f(")
        code_sample = code_sample.replace(f"{func_name}(", "f(")
    reverse_prompt = "\n\n# write the docstring for the above function\n"
    without_ref = (
        func_context
        + "\n"
        + func_header.strip()
        + "\n"
        + code_sample
        + reverse_prompt
        + func_header.strip()
        + "\n"
        + f"    {doc_start}"
    )
    with_ref = without_ref + docstring.strip()[3:]
    return with_ref.rstrip(), without_ref


def rindex(lst, value):
    return len(lst) - lst[::-1].index(value) - 1


def find_start(tokens):
    tokens = tokens[:-2]  # remove last docstring marker
    for marker in [' """', " '''", ' ""', "''"]:
        if marker in tokens:
            return rindex(tokens[:-1], marker) + 1
    raise ValueError("not found")


def batch_query_reverse_logp(all_codex_data, args):
    for outer_i, batch_start in enumerate(
        range(0, len(all_codex_data), args.batch_size)
    ):
        batch_data = all_codex_data[batch_start : batch_start + args.batch_size]
        batch_prompts = []
        batch_data_with_prompt = []
        for codex_data, problem in batch_data:
            # TODO: postprocessing, should move else where
            codex_data["trg_prediction"], codex_data["tokens"] = postprocess_func_only(
                codex_data["trg_prediction"], codex_data["tokens"]
            )
            codex_data["logprobs"] = codex_data["logprobs"][: len(codex_data["tokens"])]

            with_ref_prompt, without_ref_prompt = make_new_context(
                codex_data,
                problem,
                canonicalize=args.canonicalize,
                clean_print=args.clean_print,
            )
            batch_prompts.append(with_ref_prompt)
            batch_data_with_prompt.append(
                (codex_data, problem, with_ref_prompt, without_ref_prompt)
            )

        with_ref_reponse, _ = safe_codex_call(
            args,
            batch_prompts,
            temperature=1.0,
            echo=True,
            max_tokens=0,
            api_i=(outer_i % 3),
        )
        for (
            batch_i,
            (codex_data, problem, with_ref_prompt, without_ref_prompt),
        ) in enumerate(batch_data_with_prompt):
            num_api_tokens = find_start(
                with_ref_reponse["choices"][batch_i]["logprobs"]["tokens"]
            )
            gt_prompt_logprob = with_ref_reponse["choices"][batch_i]["logprobs"][
                "token_logprobs"
            ][num_api_tokens:]
            gt_prompt_tokens = with_ref_reponse["choices"][batch_i]["logprobs"][
                "tokens"
            ][num_api_tokens:]
            codex_data["reverse_prompt_with_ref"] = with_ref_prompt
            codex_data["reverse_prompt_without_ref"] = without_ref_prompt
            codex_data["prompt_reverse_logprobs"] = gt_prompt_logprob
            codex_data["prompt_reverse_tokens"] = gt_prompt_tokens
            codex_data["prompt_reverse_full_tokens"] = with_ref_reponse["choices"][
                batch_i
            ]["logprobs"]["tokens"]
            codex_data["prompt_reverse_full_logprobs"] = with_ref_reponse["choices"][
                batch_i
            ]["logprobs"]["token_logprobs"]
    all_codex_data = [d[0] for d in all_codex_data]
    return all_codex_data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="codex001")
    parser.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
        choices=["humaneval", "codet_humaneval", "mbpp_sanitized"],
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_procs", type=int, default=40)
    parser.add_argument(
        "--data_path",
        type=str,
        default="./samples/codex002",
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--canonicalize", default=False, action="store_true")
    parser.add_argument("--clean-print", default=False, action="store_true")
    parser.add_argument("--overwrite-output-dir", default=False, action="store_true")

    args = parser.parse_args()
    args.data_path = Path(args.data_path)
    out_dir = f"seed-*/**/*-{args.temperature}"
    if args.top_p != 1.0:
        out_dir += f"-p{args.top_p}"
    if args.max_tokens != 512:
        out_dir += f"-max{args.max_tokens}"
    args.data_path = args.data_path / args.dataset / out_dir
    paths = list(sorted(glob(str(args.data_path), recursive=True)))

    if args.dataset == "codet_humaneval":
        dataset = HumanEvalDataset(
            "dataset/human_eval/dataset/CodeTHumanEval.jsonl", mode="prompt_only"
        )
    else:
        dataset = HumanEvalDataset(
            path="dataset/mbpp/mbpp_sanitized_for_code_generation.jsonl",
            mode="prompt_only",
        )
    prompt_to_data = {p["prompt"]: p for task_id, p in dataset.raw_data.items()}

    paths = sorted(paths)
    for path in tqdm(paths, desc="total seeds", disable=False):
        path = Path(path)
        for sample_i in trange(args.num_samples):
            if len(args.tag) == 0:
                output_file_name = f"{args.split}-{sample_i}.jsonl"
            else:
                output_file_name = f"{args.split}-{sample_i}-{args.tag}.jsonl"

            try:
                all_codex_data = []
                with open(path / f"{args.split}-{sample_i}.jsonl", "r") as f:
                    for i, line in enumerate(f):
                        codex_data = json.loads(line)
                        raw_data = prompt_to_data[codex_data["prompt"]]
                        all_codex_data.append((codex_data, raw_data))
            except Exception as e:
                print(e)
                print(f"{path / output_file_name} not ready yet. skipping.")
                continue

            if (path / output_file_name).exists() and not args.overwrite_output_dir:
                with open(path / output_file_name, "r") as f:
                    line_num = len(f.readlines())
                if line_num == len(all_codex_data):
                    print(f"skipping {path / output_file_name}")
                    continue

            from multiprocessing import Pool

            if args.num_procs > 1:
                all_codex_data_with_reverse = []
                chunk_size = len(all_codex_data) // args.num_procs + 1
                chunked_all_codex_data = [
                    all_codex_data[chunk_start : chunk_start + chunk_size]
                    for chunk_start in range(0, len(all_codex_data), chunk_size)
                ]
                with Pool(processes=args.num_procs) as pool:
                    for codex_data_with_reverse in pool.imap(
                        partial(batch_query_reverse_logp, args=args),
                        chunked_all_codex_data,
                    ):
                        all_codex_data_with_reverse.extend(codex_data_with_reverse)
            else:
                all_codex_data_with_reverse = batch_query_reverse_logp(
                    all_codex_data, args
                )

            with open(path / output_file_name, "w") as f:
                for codex_data_with_reverse in all_codex_data_with_reverse:
                    codex_data_json = json.dumps(codex_data_with_reverse)
                    f.write(codex_data_json + "\n")
