import asyncio
import os
import sys
from functools import partial
from itertools import product

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import orjson
import torch
from nnsight import LanguageModel
from sae import Sae as EleutherSae
from sae import SaeConfig as EleutherSaeConfig
from sae_auto_interp.autoencoders.OpenAI.model import ACTIVATIONS_CLASSES, TopK
from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents
from sae_auto_interp.clients import Offline
from sae_auto_interp.config import CacheConfig, ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer
from sae_auto_interp.features import FeatureCache, FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import DetectionScorer, FuzzingScorer
from sae_auto_interp.utils import load_tokenized_data

import wandb
from jacobian_saes.sae_pair import SAEPair
from jacobian_saes.utils import default_device

# Hyperparams
small_llama = False
batch_size = 8
dataset_repo = "EleutherAI/rpj-v2-sample"
# dataset_repo = "stas/c4-en-10k" #! Remove
# dataset_row = "text"  #! Remove
dataset_row = "raw_content"
ctx_len = 256
example_ctx_len = 32
n_tokens = 1_000_000
# n_tokens = 10_000 #! Remove
scoring_batch_size = 5
min_examples = 200
# min_examples = 20 #! Remove
max_examples = 10000
# max_examples = 100 #! Remove
n_examples_test = 20
n_quantiles = 10
number_of_parallel_latents = 10
n_features = 512
n_layers = 6
split_percentage = "1%"

api = wandb.Api()


def load_model_and_saes():
    """Port my JSAEs into Eleuther's SAE format"""
    model = LanguageModel(
        "EleutherAI/pythia-70m-deduped",
        device_map=default_device,
        dispatch=True,
        torch_dtype="float16",
    )

    submodules = {}
    for layer in range(n_layers):
        wandb_artifact_path = f"lucyfarnik/jsaes_pythia70m1/sae_pythia-70m-deduped_blocks.{layer}.ln2.hook_normalized_16384:v0"
        local_path = "artifacts/" + wandb_artifact_path.split("/")[-1]

        # Download JSAE if not already cached
        if not os.path.exists(local_path):
            api.artifact(wandb_artifact_path).download()

        sae_pair = SAEPair.load_from_pretrained(local_path, device=default_device)
        sae_pair.half()

        eleuther_sae_cfg = EleutherSaeConfig(
            expansion_factor=sae_pair.cfg.d_sae // sae_pair.cfg.d_in,
            normalize_decoder=True,  # TODO assert that the sae_pair has this, somehow
            num_latents=sae_pair.cfg.d_sae,
            k=sae_pair.cfg.activation_fn_kwargs["k"],
        )

        for is_output_sae in [False, True]:
            # submodule = the NNSight module whose output's we're applying the SAE to
            if is_output_sae:
                submodule = model.gpt_neox.layers[layer].mlp
            else:
                submodule = model.gpt_neox.layers[layer].post_attention_layernorm

            # Load weights into EleutherSAE
            sae = EleutherSae(sae_pair.cfg.d_in, eleuther_sae_cfg, default_device)
            sae.encoder.weight.data = sae_pair.get_W_enc(is_output_sae).permute((1, 0))
            sae.encoder.bias.data = sae_pair.get_b_enc(is_output_sae)
            sae.W_dec.data = sae_pair.get_W_dec(is_output_sae)
            sae.b_dec.data = sae_pair.get_b_dec(is_output_sae)

            # Build the AutoencoderLatents wrapper (which returns the SAE latents)
            def _forward(sae, k, x):
                encoded = sae.pre_acts(x)
                if k is not None:
                    trained_k = k
                else:
                    trained_k = sae.cfg.k
                topk = TopK(trained_k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
                return topk(encoded)

            _forward = partial(_forward, sae, eleuther_sae_cfg.k)

            submodule.ae = AutoencoderLatents(sae, _forward, width=sae_pair.cfg.d_sae)

            submodules[submodule.path] = submodule
    del sae_pair, sae

    # modify the model to always apply the SAE to the module outputs
    #! TODO this is probably introducing a bug, I think we're replacing all the SAEs
    #! at the same time so downstream SAEs will underperform due to upstream reconstruction errors
    with model.edit(" ") as model:
        for _, submodule in submodules.items():
            acts = submodule.output
            submodule.ae(acts, hook=True)
    del submodule, acts

    return model, submodules, eleuther_sae_cfg


# === Cache the activations ===
def cache_acts(model, submodules):
    # There is a default cache config that can also be modified when using a "production" script.
    cfg = CacheConfig(
        dataset_repo=dataset_repo,
        dataset_split=f"train[:{split_percentage}]",
        batch_size=batch_size,
        ctx_len=ctx_len,
        n_tokens=n_tokens,
        n_splits=5,
        # n_splits=1, #! Remove
        dataset_row=dataset_row,
    )

    tokens = load_tokenized_data(
        ctx_len=cfg.ctx_len,
        tokenizer=model.tokenizer,
        dataset_repo=cfg.dataset_repo,
        dataset_split=cfg.dataset_split,
        dataset_row=cfg.dataset_row,
    )
    # Tokens should have the shape (n_batches,ctx_len)

    cache = FeatureCache(
        model,
        submodules,
        batch_size=cfg.batch_size,
    )

    cache.run(cfg.n_tokens, tokens)

    cache.save_splits(
        n_splits=cfg.n_splits,  # We split the activation and location indices into different files to make loading faster
        save_dir="latents",
    )

    # The config of the cache should be saved with the results such that it can be loaded later.

    cache.save_config(
        save_dir="latents", cfg=cfg, model_name="EleutherAI/pythia-70m-deduped"
    )


def run_autointerp_on_cached_acts(eleuther_sae_cfg):
    feature_cfg = FeatureConfig(
        width=eleuther_sae_cfg.num_latents,  # The number of latents of your SAE
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

    client = Offline(
        f"hugging-quants/Meta-Llama-3.1-{8 if small_llama else 70}B-Instruct-AWQ-INT4",
        max_memory=0.8,
        max_model_len=5120,
        num_gpus=1,
    )

    for layer, is_output_sae in product(range(n_layers), [False, True]):
        print(
            f"\nStarted running autointerp on layer {layer} {'post' if is_output_sae else 'pre'}-MLP\n\n"
        )
        module = f".gpt_neox.layers.{layer}.{"mlp" if is_output_sae else "post_attention_layernorm"}"  # The layer to score
        feature_dict = {module: torch.randperm(feature_cfg.width)[:n_features]}

        dataset = FeatureDataset(
            raw_dir="latents",  # The folder where the cache is stored
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
            with open(
                f"results/explanations/{module}/{result.record.feature}.txt",
                "wb",
            ) as f:
                f.write(orjson.dumps(result.explanation))

            return result

        # try making the directory if it doesn't exist
        os.makedirs(f"results/explanations/{module}", exist_ok=True)

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
            with open(f"results/scores/{score_dir}/{result.record.feature}.txt", "wb") as f:
                f.write(orjson.dumps(result.score))

        os.makedirs("results/scores/detection", exist_ok=True)
        os.makedirs("results/scores/fuzz", exist_ok=True)

        scorer_pipe = Pipe(
            process_wrapper(
                DetectionScorer(
                    client,
                    tokenizer=dataset.tokenizer,
                    batch_size=scoring_batch_size,
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
                    batch_size=scoring_batch_size,
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


if __name__ == "__main__":
    model, submodules, eleuther_sae_cfg = load_model_and_saes()
    print("\n=== Finished loading model and SAEs ===\n\n")

    cache_acts(model, submodules)
    del model, submodules
    print("\n=== Finished caching activations ===\n\n")

    run_autointerp_on_cached_acts(eleuther_sae_cfg)
