# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import copy
import json
import openai
import os
import pickle
import random
import signal
import time
from glob import glob
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm, trange
import re

codex_name_mapping = {
    "codex-cushman": "code-cushman-001",
    "codex002": "code-davinci-002",
    "codex001": "code-davinci-001",
}


def codex_greedy(configs, prompt, max_tokens=512):
    response = openai.Completion.create(
        engine=codex_name_mapping[configs.engine_name]
        if configs.engine_name is not None
        else "davinci-codex",
        prompt=prompt,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=configs.end_template,
    )
    return response["choices"][0]["text"], None, None


def codex_sample(configs, prompt, max_tokens=512):
    response = openai.Completion.create(
        engine=codex_name_mapping[configs.engine_name]
        if configs.engine_name is not None
        else "davinci-codex",
        prompt=prompt,
        temperature=configs.temperature,
        max_tokens=max_tokens,
        top_p=configs.top_p,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=1,
        stop=configs.end_template,
    )
    return (
        response["choices"][0]["text"],
        response["choices"][0]["logprobs"]["tokens"],
        response["choices"][0]["logprobs"]["token_logprobs"],
    )


def codex_batch_greedy(configs, batch_prompts, max_tokens=512):
    raise NotImplementedError


def codex_batch_sample(configs, batch_prompts, max_tokens=512):
    response = openai.Completion.create(
        engine=codex_name_mapping[configs.engine_name]
        if configs.engine_name is not None
        else "davinci-codex",
        prompt=batch_prompts,
        temperature=configs.temperature,
        max_tokens=max_tokens,
        top_p=configs.top_p,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=1,
        stop=configs.end_template,
    )
    return [
        (
            response["choices"][batch_i]["text"],
            response["choices"][batch_i]["logprobs"]["tokens"],
            response["choices"][batch_i]["logprobs"]["token_logprobs"],
        )
        for batch_i in range(len(batch_prompts))
    ]


def process_batch_examples(args_with_idx):
    batch_i, batch_args = args_with_idx
    all_prompts = []
    for args in batch_args:
        src, trg, info, prompt_prefix, configs = args
        if configs.dataset in ["mbpp_sanitized", "humaneval", "codet_humaneval"]:
            prompt = src
        else:
            prompt = prompt_prefix + configs.example_template.format(src=src, info=info)
        all_prompts.append(prompt)
    max_tokens = configs.max_tokens
    while True:
        if configs.engine_name == "codex002":
            openai.organization = os.getenv(f"OPENAI_ORG{(batch_i%3)+1}")
        else:
            openai.organization = os.getenv("OPENAI_ORG1")
        try:
            batch_results = (
                codex_batch_greedy(configs, all_prompts, max_tokens)
                if configs.mode == "greedy"
                else codex_batch_sample(configs, all_prompts, max_tokens)
            )
            break
        except openai.error.InvalidRequestError as e:
            print(f"Context len: halving gen tokens, curr: {max_tokens}", end="\r")
            max_tokens = max_tokens // 2
            if max_tokens < 32:
                raise ValueError("Prompt too long")
        except openai.error.RateLimitError as e:
            print(type(e), re.search("Current: .+ / min", str(e))[0], end="\r")
            time.sleep(30)
        except Exception as e:
            print(type(e), e)
            time.sleep(10)

    all_results = []
    for args, prompt, (trg_prediction, tokens, logprobs) in zip(
        batch_args, all_prompts, batch_results
    ):
        src, trg, info, prompt_prefix, configs = args
        if "humaneval" in configs.dataset or configs.dataset == "mbpp_sanitized":
            if "\nprint" in trg_prediction:
                for i in range(0, len(tokens) - 1):
                    if tokens[i : i + 2] == ["\n", "print"]:
                        break
                tokens = tokens[:i]
                logprobs = logprobs[:i]
                trg_prediction = "".join(tokens)
                if i == len(tokens) - 1:
                    raise ValueError("not matched")
        result = {
            "prompt": prompt,
            "src": src,
            "trg_prediction": trg_prediction,
            "reference": trg,
            "tokens": tokens,
            "logprobs": logprobs,
        }
        all_results.append(json.dumps(result))
    return all_results


def process_one_example(args):
    src, trg, info, prompt_prefix, configs = args
    if configs.dataset in ["mbpp_sanitized", "humaneval", "codet_humaneval"]:
        prompt = src
    else:
        prompt = prompt_prefix + configs.example_template.format(src=src, info=info)
    max_tokens = configs.max_tokens
    while True:
        try:
            trg_prediction, tokens, logprobs = (
                codex_greedy(configs, prompt, max_tokens)
                if configs.mode == "greedy"
                else codex_sample(configs, prompt, max_tokens)
            )
            break
        except openai.error.InvalidRequestError as e:
            print(f"Context len: halving gen tokens, curr: {max_tokens}", end="\r")
            max_tokens = max_tokens // 2
            if max_tokens < 32:
                raise ValueError("Prompt too long")
        except openai.error.RateLimitError as e:
            print(type(e), re.search("Current: .+ / min", str(e))[0], end="\r")
            time.sleep(30)
        except Exception as e:
            print(type(e), e)
            time.sleep(10)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            bleu_score = sentence_bleu(
                [[ch for ch in trg]], [ch for ch in trg_prediction]
            )
        except:
            bleu_score = 0
    if "humaneval" in configs.dataset or configs.dataset == "mbpp_sanitized":
        if "\nprint" in trg_prediction:
            for i in range(0, len(tokens) - 1):
                if tokens[i : i + 2] == ["\n", "print"]:
                    break
            tokens = tokens[:i]
            logprobs = logprobs[:i]
            trg_prediction = "".join(tokens)
            if i == len(tokens) - 1:
                raise ValueError("not matched")
    return json.dumps(
        {
            "prompt": prompt,
            "src": src,
            "trg_prediction": trg_prediction,
            "reference": trg,
            "tokens": tokens,
            "logprobs": logprobs,
            "bleu": bleu_score,
        }
    )


def codex_with_info(configs, dataset, prefixes):
    # model
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt_prefix = "".join(
        [
            configs.prompt_template.format(src=x[0], trg=x[1], info=x[2])
            for x in prefixes
        ]
    )

    # save folder
    if configs.top_p == 1:
        save_dir = f"{configs.output_path}/seed-{configs.seed}/{configs.n_prompts}-shot/{configs.mode}-{configs.temperature}"
    else:
        save_dir = f"{configs.output_path}/seed-{configs.seed}/{configs.n_prompts}-shot/{configs.mode}-{configs.temperature}-p{configs.top_p}"
    if configs.max_tokens != 512:
        save_dir += f"-max{configs.max_tokens}"
    os.system(f"mkdir -p {save_dir}")
    # save configs and prefixes
    if configs.rank == 0:
        with open(f"{save_dir}/prefixes.json", "w") as fout:
            json.dump(prefixes, fout)
            fout.close()
        with open(f"{save_dir}/configs.pkl", "wb") as fout:
            pickle.dump(configs, fout)
            fout.close()
    ofname = f"{save_dir}/{configs.split}-{configs.rank}.jsonl"
    if os.path.exists(ofname):
        return

    from multiprocessing import Pool

    all_args = []
    for (src, trg, info) in dataset:
        all_args.append((src, trg, info, prompt_prefix, configs))
    all_jsons = []
    if configs.n_procs > 1:
        all_args = [
            all_args[chunk_start : chunk_start + configs.batch_size]
            for chunk_start in range(0, len(all_args), configs.batch_size)
        ]
        with Pool(processes=configs.n_procs) as pool:
            for result_json in tqdm(
                pool.imap(process_batch_examples, enumerate(all_args)),
                total=len(all_args),
            ):
                all_jsons.extend(result_json)
    else:
        for batch_i, batch_start in enumerate(
            trange(0, len(all_args), configs.batch_size)
        ):
            batch_args = all_args[batch_start : batch_start + configs.batch_size]
            all_jsons.extend(process_batch_examples((batch_i, batch_args)))

    with open(ofname, "w") as fout:
        for jsonf in all_jsons:
            fout.write(jsonf + "\n")


""" example collector: <src, trg, info> """


class CollectorWithInfo(object):
    def __init__(self, configs, dataset):
        self.configs = configs
        self.dataset = dataset

    def __call__(self, rank, local_rank, world_size):
        configs = copy.deepcopy(self.configs)
        configs.rank = rank
        configs.gpu = local_rank
        configs.world_size = world_size
        args = []
        for seed in self.configs.seed:
            for n_prompts in self.configs.n_prompts:
                args.append((seed, n_prompts, configs.temperature))
        for seed, n_prompts, temperature in tqdm(args):
            configs.n_prompts = n_prompts
            configs.seed = seed
            configs.temperature = temperature
            random.seed(configs.seed)
            if configs.n_prompts == 0:
                prefixes = []
            else:
                if configs.saved_prefixes_path_template is not None:
                    prefix_pool = list()
                    for path in glob(
                        configs.saved_prefixes_path_template, recursive=True
                    ):
                        prefix_pool.extend(json.load(open(path)))
                    prefix_pool = sorted(set([tuple(x) for x in prefix_pool]))
                    prefixes = random.sample(prefix_pool, configs.n_prompts)
                else:
                    prefixes = random.sample(
                        self.dataset.data["train"], configs.n_prompts
                    )
            if configs.shuffle_prefix:
                original_prefixes = copy.deepcopy(prefixes)
                while original_prefixes == prefixes:
                    random.shuffle(prefixes)
            codex_with_info(configs, self.dataset.data[configs.split], prefixes)

    @staticmethod
    def parse_args(main_parser=None):
        if main_parser is None:
            main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers(title="commands", dest="mode")
        # collect
        parser = subparsers.add_parser("collect", help="collecting stage")
        parser.add_argument("--output-path", type=str, required=True)
        parser.add_argument(
            "--split", type=str, default="dev", choices=["train", "dev", "test"]
        )
        parser.add_argument("--seed", type=int, nargs="+", default=[0])
        parser.add_argument("--n-procs", type=int, default=1)
        parser.add_argument(
            "--n-prompts",
            type=int,
            nargs="+",
            default=[3],
            help="number of few-shot prompt examples",
        )
        parser.add_argument(
            "--mode", type=str, default="greedy", choices=["greedy", "sample"]
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=5,
            help="number of sampled examples under the sampling mode",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=512,
            help="number of sampled examples under the sampling mode",
        )
        parser.add_argument(
            "--temperature", type=float, default=0.3, help="sample temperature"
        )
        parser.add_argument(
            "--top_p", type=float, default=1.0, help="sample temperature"
        )
        parser.add_argument(
            "--prompt-template",
            type=str,
            default="<info>{info}</info>\n<text>{src}</text>\n<code>{trg}</code>\n",
        )
        parser.add_argument(
            "--example-template",
            type=str,
            default="<info>{info}</info>\n<text>{src}</text>\n<code>",
        )
        parser.add_argument("--end-template", type=str, default="</code>")
        parser.add_argument("--shuffle-prefix", action="store_true", default=False)
        parser.add_argument("--saved-prefixes-path-template", type=str, default=None)
        parser.add_argument(
            "--engine-name",
            type=str,
            default="codex-cushman",
            choices=["codex-cushman", "codex001", "codex002"],
        )
        # slurm arguments
        parser.add_argument("--slurm-ntasks", type=int, default=None)
        parser.add_argument("--slurm-ngpus", type=int, default=0)
        parser.add_argument("--slurm-nnodes", type=int, default=1)
        parser.add_argument("--slurm-partition", type=str, default="devlab")

        args = main_parser.parse_args()

        return args

    @classmethod
    def from_args(cls, args=None, dataset=None):
        if args is None:
            args = cls.parse_args()
        assert dataset is not None
        return cls(args, dataset)
