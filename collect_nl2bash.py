# Copyright (c) Meta Platforms, Inc. and affiliates.

from data import NL2BashDataset
from collectors import CollectorWithInfo
import argparse


if __name__ == "__main__":
    dataset = NL2BashDataset()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=25)
    parser.add_argument("--num_samples", type=int, default=5)
    args = CollectorWithInfo.parse_args(parser)
    args.dataset = "nl2bash"
    args.seed = list(range(args.num_seeds))
    args.prompt_template = "<text>{src}</text>\n<code>{trg}</code>\n"
    args.example_template = "<text>{src}</text>\n<code>"
    collector = CollectorWithInfo.from_args(args, dataset)
    for i in range(args.num_samples):
        collector(i, i, 5)
