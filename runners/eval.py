import argparse
from dataclasses import dataclass
import json
import os
import sys
from tqdm import tqdm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from datasets import load_dataset
import torch
from jacobian_saes.utils import load_pretrained, run_sandwich
from jacobian_saes.evals import get_recons_loss


parser = argparse.ArgumentParser(
    description="Run evaluations of a JSAE"
)
parser.add_argument(
    "--output-dir",
    "-o",
    type=str,
    default="results/eval",
    help="Output directory",
)
parser.add_argument(
    "--path",
    "-p",
    type=str,
    default="lucyfarnik/jsaes_pythia70m2/sae_pair_pythia-70m-deduped_layer3_16384_J1_k32:v0",
    help="Wandb path to the SAE pair artifact",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.01,
    help="Threshold for counting sparsity",
)
parser.add_argument(
    "--tokens",
    "-t",
    type=float,
    default=10_000_000,
    help="Total number of training tokens",
)
args = parser.parse_args()

n_tokens = int(args.tokens)


@dataclass
class DummyActivationStore:
    normalize_activations: str = "none"


dummy_activation_store = DummyActivationStore()


sae_pair, model, mlp_with_grads, layer = load_pretrained(args.path,
                                                         use_training_class=True)
k = sae_pair.cfg.activation_fn_kwargs["k"]

dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

jac_abs_above_thresh = 0.0
ce_scores = []
kl_scores = []
with torch.no_grad():
    with tqdm(total=n_tokens) as pbar:
        for item in dataset:
            _, cache = model.run_with_cache(
                item["text"],
                stop_at_layer=layer + 1,
                names_filter=[sae_pair.cfg.hook_name],
            )
            acts = cache[sae_pair.cfg.hook_name]
            jacobian, acts_dict = run_sandwich(sae_pair, mlp_with_grads, acts)

            jac_abs_above_thresh += (jacobian.abs() > args.threshold).sum().item()

            recons_dict = get_recons_loss(
                sae_pair,
                model,
                item["text"],
                dummy_activation_store,
                compute_kl=True,
                compute_ce_loss=True,
            )

            ce_scores.append(((
                recons_dict["ce_loss_with_ablation"] - (
                    (recons_dict["ce_loss_with_sae"] +
                     recons_dict["ce_loss_with_sae2"]) / 2)
            ) / (recons_dict["ce_loss_with_ablation"] -
                 recons_dict["ce_loss_without_sae"])).flatten())
            
            kl_scores.append(((
                recons_dict["kl_div_with_ablation"] - (
                    (recons_dict["kl_div_with_sae"] +
                     recons_dict["kl_div_with_sae2"]) / 2)
            ) / recons_dict["kl_div_with_ablation"]).flatten())

            pbar.update(acts.shape[1])
            if pbar.n >= n_tokens:
                break

output_dict = {
    "jac_abs_above_thresh": jac_abs_above_thresh / pbar.n,
    "thresh": args.threshold,
    "ce_score": torch.cat(ce_scores).mean().item(),
    "kl_score": torch.cat(kl_scores).mean().item(),
    "path": args.path,
    "tokens": n_tokens,
}

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(script_dir, args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, f"{args.path.split("/")[-1]}.safetensor")

with open(output_path, 'w') as f:
    json.dump(output_dict, f)

