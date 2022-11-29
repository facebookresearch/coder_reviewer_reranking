# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

import data
from collectors import CollectorWithInfo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--info-mode",
        type=str,
        default="assertion",
        choices=["function_name", "assertion"],
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="MBPPGoogleDataset",
        choices=["MBPPDataset", "MBPPGoogleDataset"],
    )
    parser.add_argument("--num_seeds", type=int, default=25)
    parser.add_argument("--num_samples", type=int, default=5)
    args = CollectorWithInfo.parse_args(parser)
    args.seed = list(range(args.num_seeds))
    args.dataset = "mbpp"
    args.split = "test"
    dataset = getattr(data, args.dataset_type)(mode=args.info_mode)
    collector = CollectorWithInfo.from_args(args, dataset)
    for i in range(args.num_samples):
        collector(i, i, 5)
