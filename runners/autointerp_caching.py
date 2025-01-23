# Stage 1 of autointerp

from functools import partial
import os
import sys
import wandb

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from nnsight import LanguageModel
from sae import Sae as EleutherSae
from sae import SaeConfig as EleutherSaeConfig
from sae_auto_interp.autoencoders.OpenAI.model import ACTIVATIONS_CLASSES, TopK
from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data

from jacobian_saes.sae_pair import SAEPair
from jacobian_saes.utils import default_device

# Hyperparams
batch_size = 8
dataset_repo = "EleutherAI/rpj-v2-sample"
dataset_row = "raw_content"
ctx_len = 256
n_tokens = 1_000_000
n_layers = 6
split_percentage = "1%"
output_dir = "latents"
model_name = "EleutherAI/pythia-70m-deduped"

api = wandb.Api()

model = LanguageModel(
    model_name,
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
with model.edit(" ") as model:
    for _, submodule in submodules.items():
        acts = submodule.output
        submodule.ae(acts, hook=True)
del submodule, acts

# === Cache the activations ===
# There is a default cache config that can also be modified when using a "production" script.
cfg = CacheConfig(
    dataset_repo=dataset_repo,
    dataset_split=f"train[:{split_percentage}]",
    batch_size=batch_size,
    ctx_len=ctx_len,
    n_tokens=n_tokens,
    n_splits=5,
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

cache = FeatureCache(model, submodules, batch_size=cfg.batch_size)

cache.run(cfg.n_tokens, tokens)

# We split the activation and location indices into different files to make loading faster
cache.save_splits(n_splits=cfg.n_splits, save_dir=output_dir)

# The config of the cache should be saved with the results such that it can be loaded later.

cache.save_config(save_dir=output_dir, cfg=cfg, model_name=model_name)
