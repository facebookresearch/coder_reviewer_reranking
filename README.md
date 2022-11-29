# Coder Reviewer Reranking for Code Generation
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Official code release for the paper [Coder Reviewer Reranking for Code Generation](www.arxiv.com).


## Setup
### Downloading data and cached outputs
1. For convenience, we include data used for this project in [`dataset.zip`](https://dl.fbaipublicfiles.com/coder-reviewer/dataset.zip). You need to download and unzip this file before using this repo.
These include 
- [HumanEval](https://github.com/openai/human-eval). We also include the prompt used in the [CodeT](https://github.com/microsoft/CodeT/tree/main/CodeT) paper
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp), which includes both the sanitized version and the initial version.
- [Spider](https://github.com/taoyds/spider) includes the evaluation script and the data. We also include the cached outputs from executing the groundtruth SQL queries.
- [NL2BASH](https://github.com/TellinaTool/nl2bash/tree/master/data)
2. Samples and precomputed execution results can be found in [`samples.zip`](https://dl.fbaipublicfiles.com/coder-reviewer/samples.zip)

### Installing software environment
1. All experiments are run with `python==3.8.13`. 
2. Install [pyminifier](https://github.com/liftoff/pyminifier/tree/master) from source.
Installing `pyminifier` requires reverting setup tools to an older version (`pip install setuptools==57.5.0`). 
For other issues of installing `pyminifier`, checkout their [issues](https://github.com/liftoff/pyminifier/issues) for potential fixes.
3. Install `torch==1.12.1`. You should install a distribution that matches your hardware environment 
4. Install the other packages by 
```bash
pip install -r requirements.txt
```

## Usage

### Running the selector with released outputs
1. We release samples obtained from the OpenAI codex API in `samples.zip`. Unzipping this file, you should see a folder with the below structure
```bash
samples
├── codex-cushman
│   ├── codet_humaneval
│   └── mbpp_sanitized
├── codex001
└── codex002
```
We will go over the code/commands you need to collect these samples in a later section.
2. Run the following script to compare different reranking methods.
```bash
model="codex002"
dataset="mbpp_sanitized"
outdir="result_db"
python sample_selectors.py --model ${model} \
    --num_samples_end 25 \
    --num_samples_gap 5 \
    --data_path samples \
    --out_dir ${outdir} \
    --dataset ${dataset} \
    --num_procs 10 \
    --num_bootstraps 50 \
    --temperature 0.4 \
    --verbose\
```
3. We have included the execution results of all generated samples in the `samples.zip`. If you want to execute the generated programs yourself, you can run the following command. Typically, we leverage aggressive multiprocessing to speed up this process. You can change the number of processes by modifying `nprocs`.
Modify the `model` and `dataset` arguments to execute other models and datasets. 
```bash
model="codex002"
dataset="codet_humaneval"
nprocs=25
torchrun --nproc_per_node=${nprocs} multi_exec.py --temperature 0.4 --world_size 25 --dataset ${dataset} --in_data_path samples/${model} --batch_size 4 --num_seeds 25 --num_samples 5 --num_prompts 0
```

The outputs will look like and a dictionary object containing the result will be saved into `result_db`
```
sum_logprob 0.5587 0.01
avg_logprob 0.5832 0.01
avg_reverse_logprob 0.5626 0.01
random 0.5562 0.01
sumreverselogprob-ensemble#0.5 0.6152 0.01
avgreverselogprob-ensemble#0.5 0.5963 0.01
executability-sum_logprob 0.5976 0.01
executability-avg_logprob 0.6049 0.01
executability-avg_reverse_logprob 0.5952 0.01
executability-random 0.5881 0.01
executability-sumreverselogprob-ensemble#0.5 0.6440 0.01
executability-avgreverselogprob-ensemble#0.5 0.6159 0.01
mbr_exec 0.6389 0.01
oracle 0.7891 0.01
```

### Collecting Samples
1. the below example command collects 125 (5x25) samples for zeroshot humaneval with codex002. explore `collect*.py` for collecting samples on other datasets. These scripts collect programs given the language instructions, i.e., implementing the Coder model.
```
python collect_zeroshot.py --num_samples 5 --num_seeds 25 --dataset codet_humaneval collect --output-path samples/codex002 --engine-name codex002 --temperature 0.4 --split test --n-procs 1 --batch-size 20 --mode sample --n-prompts 0
```
2. We collect the reviewer model p(instruction|generated program) by `fewshot_reviewer.py` and `zeroshot_reviewer.py`. Here's an example command for humaneval with codex002,
```
python zeroshot_reviewer.py --num_procs 1 --batch_size 20 --temperature 0.4 --num_samples 5 --split test --dataset codet_humaneval --model codex002 --data_path samples/codex002 --canonicalize --clean-print
```
This code will update the cached results with the reviewer model probability. Explore other arguments to run for different models and datasets.

#### Authors
- [Tianyi Zhang](https://tiiiger.github.io/)
- [Tao Yu](https://taoyds.github.io/)
- [Tatsunori Hashimoto](https://thashim.github.io/)
- [Mike Lewis](https://research.facebook.com/people/lewis-mike/)
- [Scott Wen-tau Yih](https://scottyih.org/)
- [Daniel Fried](https://dpfried.github.io/)
- [Sida I. Wang](http://www.sidaw.xyz/)

#### Acknowledgement
This codebase is largely adapted from [MBR-Exec](https://github.com/facebookresearch/mbr-exec).

#### License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png

#### Citation
If you find our work helpful, please cite as
```
@article{Zhang2022Coder,
  title={Coder Reviewer Reranking for Code Generation},
  author={Tianyi Zhang and Tao Yu and Tatsunori B. Hashimoto and Mike Lewis and Wen-tau Yih and Daniel Fried and Sida I. Wang},
  journal={ArXiv},
  year={2022},
  volume={abs/}
}
```
