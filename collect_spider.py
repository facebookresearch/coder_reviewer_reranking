# Copyright (c) Meta Platforms, Inc. and affiliates.

from data import SpiderDataset
from collectors import CollectorWithInfo
import argparse


if __name__ == "__main__":
    dataset = SpiderDataset()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=25)
    parser.add_argument("--num_samples", type=int, default=5)
    args = CollectorWithInfo.parse_args(parser)
    args.dataset = "spider"
    args.seed = list(range(args.num_seeds))
    collector = CollectorWithInfo.from_args(args, dataset)
    for i in range(args.num_samples):
        collector(i, i, 5)
