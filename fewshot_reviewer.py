# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path
import os
from glob import glob
from argparse import ArgumentParser
import html
import json
from utils import *
from tqdm import tqdm, trange
from functools import partial
from utils import write_jsonl, parse_prompt, make_new_context
from pyminifier_canonicalize import remove_print, clean_comment

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="codex")
parser.add_argument(
    "--dataset", type=str, default="mbpp", choices=["mbpp", "spider", "nl2bash"]
)
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--max_tokens", type=int, default=512)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--num_procs", type=int, default=40)
parser.add_argument("--canonicalize", action="store_true", default=False)
parser.add_argument(
    "--data_path",
    type=str,
    default="/private/home/tianyizzz/projects/mbr-exec-data/mbr-exec-release/",
)
parser.add_argument("--temperature", type=float, default=0.3)

args = parser.parse_args()
args.data_path = Path(args.data_path)
out_dir = f"seed-*/**/*-{args.temperature}"
if args.top_p != 1.0:
    out_dir += f"-p{args.top_p}"
if args.max_tokens != 512:
    out_dir += f"-max{args.max_tokens}"
args.data_path = args.data_path / args.dataset / out_dir
paths = list(sorted(glob(str(args.data_path), recursive=True)))


def find_start(tokens, dataset="mbpp"):
    if dataset == "mbpp":
        match_token = ["<", "info", ">"]
    else:
        match_token = ["<", "text", ">"]
    for i in range(len(tokens) - 3, 0, -1):
        if tokens[i : i + 3] == match_token:
            break
    return i


def batch_query_reverse_logp(all_codex_data, args, verbose=False):
    for outer_i, batch_start in enumerate(
        trange(0, len(all_codex_data), args.batch_size, disable=not verbose)
    ):
        batch_data = all_codex_data[batch_start : batch_start + args.batch_size]
        batch_prompts = []
        batch_prompts_without_ref = []
        for codex_data in batch_data:
            prompt = codex_data["prompt"]
            prompt_parse = parse_prompt(prompt, dataset=args.dataset)
            code_sample = codex_data["trg_prediction"]
            prompt_parse[-1]["code"] = f"<code>{code_sample}</code>"
            if args.dataset == "mbpp" and args.canonicalize:
                try:
                    code_sample = clean_comment(code_sample)
                except:
                    code_sample = code_sample
                code_sample = remove_print(code_sample)
            with_ref_prompt, without_ref_prompt = make_new_context(
                prompt_parse, dataset=args.dataset
            )
            batch_prompts.append(with_ref_prompt)
            batch_prompts_without_ref.append(without_ref_prompt)
        with_ref_reponse, _ = safe_codex_call(
            args,
            batch_prompts,
            temperature=1.0,
            echo=True,
            max_tokens=0,
            api_i=outer_i % 3,
        )
        for batch_i, (codex_data, with_ref_prompt, without_ref_prompt) in enumerate(
            zip(batch_data, batch_prompts, batch_prompts_without_ref)
        ):
            num_api_tokens = find_start(
                with_ref_reponse["choices"][batch_i]["logprobs"]["tokens"],
                dataset=args.dataset,
            )
            gt_prompt_logprob = with_ref_reponse["choices"][batch_i]["logprobs"][
                "token_logprobs"
            ][num_api_tokens:]
            gt_prompt_tokens = with_ref_reponse["choices"][batch_i]["logprobs"][
                "tokens"
            ][num_api_tokens:]
            codex_data["reverse_prompt_with_ref"] = with_ref_prompt
            codex_data["reverse_prompt_without_ref"] = without_ref_prompt
            codex_data["prompt_reverse_tokens"] = gt_prompt_tokens
            codex_data["prompt_reverse_logprobs"] = gt_prompt_logprob
            codex_data["prompt_reverse_full_tokens"] = with_ref_reponse["choices"][
                batch_i
            ]["logprobs"]["tokens"]
            codex_data["prompt_reverse_full_logprobs"] = with_ref_reponse["choices"][
                batch_i
            ]["logprobs"]["token_logprobs"]
    return all_codex_data


paths = sorted(paths)
print(paths)
for path in tqdm(paths, desc="total seeds", disable=False):
    path = Path(path)
    for sample_i in trange(args.num_samples, leave=False):
        if len(args.tag) == 0:
            output_file_name = f"{args.split}-{sample_i}-with-reverse.jsonl"
        else:
            output_file_name = f"{args.split}-{sample_i}-with-reverse-{args.tag}.jsonl"

        try:
            all_codex_data = []
            with open(path / f"{args.split}-{sample_i}.jsonl", "r") as f:
                for i, line in enumerate(f):
                    codex_data = json.loads(line)
                    codex_data = json.loads(line)
                    all_codex_data.append(codex_data)
        except Exception as e:
            print(e)
            print(f"{path / output_file_name} not ready yet. skipping.")
            continue

        if (path / output_file_name).exists():
            with open(path / output_file_name, "r") as f:
                line_num = len(f.readlines())
            if line_num == len(all_codex_data):
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
                for codex_data_with_reverse in tqdm(
                    pool.imap(
                        partial(batch_query_reverse_logp, args=args, verbose=True),
                        chunked_all_codex_data,
                    ),
                    total=len(chunked_all_codex_data),
                ):
                    all_codex_data_with_reverse.extend(codex_data_with_reverse)
        else:
            all_codex_data_with_reverse = batch_query_reverse_logp(
                all_codex_data, args, verbose=True
            )

        with open(path / output_file_name, "w") as f:
            for codex_data_with_reverse in all_codex_data_with_reverse:
                codex_data_json = json.dumps(codex_data_with_reverse)
                f.write(codex_data_json + "\n")
