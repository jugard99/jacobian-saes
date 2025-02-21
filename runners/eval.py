import argparse
import os
import sys
from dataclasses import dataclass

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from jacobian_saes.evals import get_recons_loss
from jacobian_saes.utils import load_pretrained, run_sandwich
from jacobian_saes.training.sparsity_metrics import _kurtosis
from jacobian_saes.evals import jac_norms

parser = argparse.ArgumentParser(description="Run evaluations of a JSAE")
parser.add_argument(
    "--norm",
    type=str,
    choices=jac_norms.keys(),
    help="Normalization for the Jacobian",
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


sae_pair, model, mlp_with_grads, layer = load_pretrained(
    args.path, use_training_class=True
)
k = sae_pair.cfg.activation_fn_kwargs["k"]

dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)


def safe_division(numerator, denominator, eps=1e-8):
    # Add small epsilon to denominator to prevent division by zero
    # Also clip the result to handle extreme values
    result = numerator / (denominator + eps)
    result[~result.isfinite()] = 0.0
    return torch.clamp(result, min=0.0, max=2.0)


jac_abs_above_thresh = 0.0
jac_kurtosis = 0.0
norm_vals = {name: 0.0 for name in jac_norms.keys() if "t" in name}
ce_scores = []
ce_scores2 = []
kl_scores = []
kl_scores2 = []
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
            jac_kurtosis += _kurtosis(jacobian).sum().item()

            for name, norm_fn in jac_norms.items():
                if "t" not in name:
                    continue
                norm_val = norm_fn(jacobian)
                norm_vals[name] += norm_val.sum().item()

            if args.norm is not None:
                jacobian = jacobian / jac_norms[args.norm](jacobian)
            jac_abs_above_thresh += (jacobian.abs() > args.threshold).sum().item()

            recons_dict = get_recons_loss(
                sae_pair,
                model,
                item["text"],
                dummy_activation_store,
                compute_kl=True,
                compute_ce_loss=True,
            )

            ce_scores.append(
                safe_division(
                    recons_dict["ce_loss_with_ablation"]
                    - recons_dict["ce_loss_with_sae"],
                    recons_dict["ce_loss_with_ablation"]
                    - recons_dict["ce_loss_without_sae"],
                ).flatten()
            )

            ce_scores2.append(
                safe_division(
                    recons_dict["ce_loss_with_ablation2"]
                    - recons_dict["ce_loss_with_sae2"],
                    recons_dict["ce_loss_with_ablation2"]
                    - recons_dict["ce_loss_without_sae"],
                ).flatten()
            )

            kl_scores.append(
                safe_division(
                    recons_dict["kl_div_with_ablation"]
                    - recons_dict["kl_div_with_sae"],
                    recons_dict["kl_div_with_ablation"],
                ).flatten()
            )

            kl_scores2.append(
                safe_division(
                    recons_dict["kl_div_with_ablation2"]
                    - recons_dict["kl_div_with_sae2"],
                    recons_dict["kl_div_with_ablation2"],
                ).flatten()
            )

            pbar.update(acts.shape[1])
            if pbar.n >= n_tokens:
                break


def get_mean(tensors: list[torch.Tensor]) -> str:
    concatenated = torch.cat(tensors)
    mask = concatenated.isfinite()
    mean = concatenated[mask].mean().item()
    return str(mean)


tensors_dict = {
    "jac_abs_above_thresh": torch.tensor(jac_abs_above_thresh / pbar.n),
    "jac_kurtosis": torch.tensor(jac_kurtosis / pbar.n),
    "thresh": torch.tensor(args.threshold),
    "ce_scores": torch.cat(ce_scores),
    "ce_scores2": torch.cat(ce_scores2),
    "kl_scores": torch.cat(kl_scores),
    "kl_scores2": torch.cat(kl_scores2),
}
metadata = {
    "mean_ce_score": get_mean(ce_scores),
    "mean_ce_score2": get_mean(ce_scores2),
    "mean_kl_score": get_mean(kl_scores),
    "mean_kl_score2": get_mean(kl_scores2),
    "path": args.path,
    "tokens": str(n_tokens),
}

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(script_dir, args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, f"{args.path.split("/")[-1]}.safetensor")
save_file(tensors_dict, output_path, metadata=metadata)
