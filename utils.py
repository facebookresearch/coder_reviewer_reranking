# Copyright (c) Meta Platforms, Inc. and affiliates.

from time import sleep
import os
import random
import openai
import re
import json


def safe_codex_call(
    args, api_text, temperature=None, stop=None, echo=False, max_tokens=256, api_i=0
):
    temperature = temperature if temperature else args.temperature
    while True:
        try:
            if args.model == "codex002":
                openai.organization = os.getenv(f"OPENAI_ORG{api_i+1}")
            else:
                openai.organization = os.getenv("OPENAI_ORG1")
            codex_response = codex_greedy(
                api_text,
                temperature=temperature,
                codex_config=args.model,
                stop=stop,
                echo=echo,
                max_tokens=max_tokens,
            )
            break
        except openai.error.InvalidRequestError as e:
            codex_response = None
            if isinstance(api_text, list):
                api_text = [t.replace("\n", "") for t in api_text]
            else:
                api_text = api_text.replace("\n", "")
            print("Invalid Request: Removing newlines")
        except openai.error.RateLimitError as e:
            print(type(e), f"API {api_i}:", e, end="\r")
            sleep(30)
            api_i = (api_i + 1) % 3
        except Exception as e:
            print(type(e), e)
            sleep(10)

    if codex_response is None:
        codex_text = ""
    else:
        codex_text = "".join(codex_response["choices"][0]["logprobs"]["tokens"])
    return codex_response, codex_text


def codex_greedy(
    prompt, temperature=0.3, codex_config="codex", stop=None, echo=False, max_tokens=256
):
    if stop is None:
        stop = ["#SOLUTION END", "# SOLUTION END", "SOLUTION END"]
    if codex_config == "codex001":
        codex_code = "code-davinci-001"
    elif codex_config == "codex002":
        codex_code = "code-davinci-002"
    elif codex_config == "codex-cushman":
        codex_code = "code-cushman-001"
    else:
        raise ValueError

    response = openai.Completion.create(
        engine=codex_code,
        prompt=prompt,
        temperature=temperature,
        stop=stop,
        max_tokens=max_tokens,
        top_p=0.95,
        logprobs=1,
        frequency_penalty=0,
        presence_penalty=0,
        echo=echo,
    )
    return response


def write_jsonl(data_list, file_path):
    with open(file_path, "w") as f:
        for d in data_list:
            f.write(json.dumps(d) + "\n")


def parse_prompt(prompt, dataset="mbpp"):
    prompt_data = []
    fewshot_examples = [
        p.strip() + "</code>" for p in prompt.split("</code>") if len(p) > 1
    ]
    for example in fewshot_examples:
        example_data = dict()
        if dataset in ["mbpp", "spider"]:
            all_fields = ["info", "text", "code"]
        elif dataset == "nl2bash":
            all_fields = ["text", "code"]
        for field in all_fields:
            field_start = example.index(f"<{field}>")
            field_end = example.index(f"</{field}>")
            example_data[field] = example[field_start : field_end + len(f"</{field}>")]
        prompt_data.append(example_data)
    return prompt_data


def make_new_context(prompt_parse, dataset="mbpp"):
    without_ref = ""
    with_ref = ""

    if dataset == "mbpp":
        full_prompt_fields = ["code", "info", "text"]
    elif dataset == "spider":
        full_prompt_fields = ["info", "code", "text"]
    else:
        full_prompt_fields = ["code", "text"]

    if dataset == "mbpp" or dataset == "nl2bash":
        partial_prompt_fields = ["code"]
    elif dataset == "spider":
        partial_prompt_fields = ["info", "code"]

    for i, example in enumerate(prompt_parse):
        for field in full_prompt_fields:
            with_ref += example[field] + "\n"
        if i < len(prompt_parse) - 1:
            for field in full_prompt_fields:
                without_ref += example[field] + "\n"
        else:
            for field in partial_prompt_fields:
                without_ref += example[field] + "\n"
    return with_ref.strip(), without_ref.strip()


from contextlib import contextmanager
import signal


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
