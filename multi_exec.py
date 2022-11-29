# Copyright (c) Meta Platforms, Inc. and affiliates.

import shutil
import torch
from pathlib import Path
import os
from glob import glob
from argparse import ArgumentParser
from tqdm import tqdm, trange
import torch.distributed as dist
from execution import (
    execute_humaneval_folder_one,
    execute_mbpp_google_folder_one,
    execute_spider_folder_one,
    execute_nl2bash_folder_one,
)
from pathlib import Path


parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--dataset", type=str, default="mbpp")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--num_seeds", type=int, default=5)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--num_prompts", type=int, default=1)
parser.add_argument(
    "--in_data_path",
    type=str,
    default="/private/home/tianyizzz/projects/mbr-exec-data/mbr-exec-codex001/",
)
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--max_tokens", type=int, default=512)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--world_size", type=int, default=1)

args = parser.parse_args()
args.rank = int(os.environ.get("LOCAL_RANK", 0))
# if args.world_size > 1:
#     dist.init_process_group("gloo", rank=args.rank, world_size=args.world_size)

paths = []
if args.temperature > 0:
    for seed in range(args.num_seeds):
        for i in range(args.num_samples):
            if (seed * args.num_samples + i) % args.world_size == args.rank:
                out_dir = f"sample-{args.temperature}"
                if args.top_p != 1.0:
                    out_dir += f"-p{args.top_p}"
                if args.max_tokens != 512:
                    out_dir += f"-max{args.max_tokens}"
                if args.tag == "":
                    result_file = f"{args.split}-{i}.jsonl"
                else:
                    result_file = f"{args.split}-{i}-{args.tag}.jsonl"
                path = (
                    Path(args.in_data_path)
                    / args.dataset
                    / f"seed-{seed}"
                    / f"{args.num_prompts}-shot"
                    / out_dir
                    / result_file
                )
                paths.append(path)
else:
    for seed in range(args.num_seeds):
        i = 0
        if (seed * 5 + i) % args.world_size == args.rank:
            out_dir = f"sample-{args.temperature}"
            if args.max_tokens != 512:
                out_dir += f"-max{args.max_tokens}"
            if args.tag == "":
                result_file = f"{args.split}-{i}.jsonl"
            else:
                result_file = f"{args.split}-{i}-{args.tag}.jsonl"
            paths.append(
                Path(args.in_data_path)
                / args.dataset
                / f"seed-{seed}"
                / f"{args.num_prompts}-shot"
                / out_dir
                / result_file
            )

for path in tqdm(paths, disable=not args.rank == 0):
    if args.dataset == "mbpp":
        execute_mbpp_google_folder_one(path, verbose=args.rank == 0, tag=args.tag)
    elif args.dataset == "spider":
        execute_spider_folder_one(path, verbose=args.rank == 0, tag=args.tag)
    elif "humaneval" in args.dataset or args.dataset == "mbpp_sanitized":
        execute_humaneval_folder_one(
            path, verbose=args.rank == 0, tag=args.tag, dataset_choice=args.dataset
        )
    elif args.dataset == "nl2bash":
        execute_nl2bash_folder_one(path)

    else:
        raise ValueError("invalid dataset")
