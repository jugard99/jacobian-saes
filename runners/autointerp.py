# Stage 2 of autointerp: you already have the cached latents from `autointerp_caching`, this runs the actual autointerp

import argparse
import asyncio
import os
import sys
from functools import partial

import orjson
import torch

import wandb

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from sae_auto_interp.clients import Offline
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import DetectionScorer, FuzzingScorer

parser = argparse.ArgumentParser(
    description="Run autointerp on an SAE (note that you need to have run the caching script first)"
)
parser.add_argument(
    "--layer",
    "-l",
    type=int,
    default=3,
    help="Layer of the LLM",
)
parser.add_argument(
    "--is-output-sae",
    "-o",
    action='store_true',
    help="Whether the layer is the output of the SAE",
)
args = parser.parse_args()

# Hyperparams
small_llama = False
batch_size = 5
example_ctx_len = 32
min_examples = 200
max_examples = 10000
n_examples_test = 20
n_quantiles = 10
number_of_parallel_latents = 10
n_features = 512
sae_width = 16_384
latents_dir = "latents"
output_dir = "results/autointerp"

api = wandb.Api()

client = Offline(
    f"hugging-quants/Meta-Llama-3.1-{8 if small_llama else 70}B-Instruct-AWQ-INT4",
    max_memory=0.8,
    max_model_len=5120,
    num_gpus=1,
)

feature_cfg = FeatureConfig(
    width=sae_width,  # The number of latents of your SAE
    min_examples=min_examples,  # The minimum number of examples to consider for the feature to be explained
    max_examples=max_examples,  # The maximum number of examples to be sampled from
    n_splits=5,  # How many splits was the cache split into
)

experiment_cfg = ExperimentConfig(
    n_examples_test=n_examples_test,  # Number of examples to sample for testing
    n_quantiles=n_quantiles,  # Number of quantiles to sample
    example_ctx_len=example_ctx_len,  # Length of each example
    test_type="quantiles",  # Type of sampler to use for testing.
)

module = f".gpt_neox.layers.{args.layer}.{"mlp" if args.is_output_sae else "post_attention_layernorm"}"  # The layer to score
feature_dict = {module: torch.randperm(feature_cfg.width)[:n_features]}

dataset = FeatureDataset(
    raw_dir=latents_dir,  # The folder where the cache is stored
    cfg=feature_cfg,
    modules=[module],
    features=feature_dict,
)

constructor = partial(
    default_constructor,
    tokens=dataset.tokens,
    n_random=experiment_cfg.n_random,
    ctx_len=experiment_cfg.example_ctx_len,
    max_examples=feature_cfg.max_examples,
)
sampler = partial(sample, cfg=experiment_cfg)
loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)


def explainer_postprocess(result):
    with open(f"{output_dir}/explanations/{module}/{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.explanation))

    return result


# try making the directory if it doesn't exist
os.makedirs(f"{output_dir}/explanations/{module}", exist_ok=True)

explainer_pipe = process_wrapper(
    DefaultExplainer(
        client, tokenizer=dataset.tokenizer, threshold=0.3, activations=True
    ),
    postprocess=explainer_postprocess,
)


# Builds the record from result returned by the pipeline
def scorer_preprocess(result):
    record = result.record
    record.explanation = result.explanation
    record.extra_examples = record.random_examples

    return record


# Saves the score to a file
#! add module (and prob also model and SAE names) to folder path
def scorer_postprocess(result, score_dir):
    feature_idx = result.record.feature.feature_index
    with open(f"{output_dir}/scores/{score_dir}/{module}/{feature_idx}.json", "wb") as f:
        f.write(orjson.dumps(result.score))


os.makedirs(f"{output_dir}/scores/detection/{module}", exist_ok=True)
os.makedirs(f"{output_dir}/scores/fuzz/{module}", exist_ok=True)

scorer_pipe = Pipe(
    process_wrapper(
        DetectionScorer(
            client,
            tokenizer=dataset.tokenizer,
            batch_size=batch_size,
            verbose=False,
            log_prob=True,
        ),
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir="detection"),
    ),
    process_wrapper(
        FuzzingScorer(
            client,
            tokenizer=dataset.tokenizer,
            batch_size=batch_size,
            verbose=False,
            log_prob=True,
        ),
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir="fuzz"),
    ),
)

pipeline = Pipeline(loader, explainer_pipe, scorer_pipe)
print("Setting off the pipeline")
asyncio.run(pipeline.run(number_of_parallel_latents))
