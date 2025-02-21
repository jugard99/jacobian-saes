import argparse
import math
import os
import sys

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from jacobian_saes.utils import load_pretrained, run_sandwich
from jacobian_saes.evals import jac_norms

parser = argparse.ArgumentParser(
    description="Get histogram data of the Jacobian values"
)
parser.add_argument(
    "--context-size", "-c", type=int, default=2048, help="Max context size"
)
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
    default="results/histograms",
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
    "--tokens",
    "-t",
    type=float,
    default=10_000_000,
    help="Total number of training tokens",
)
args = parser.parse_args()

assert args.norm is None or args.norm[-1] == "t", "Only Lp norms per-token are supported"

n_tokens = int(args.tokens)

sae_pair, model, mlp_with_grads, layer = load_pretrained(args.path)
k = sae_pair.cfg.activation_fn_kwargs["k"]

dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

if args.norm is not None:
    n_tokens_norm_est = 100_000
    jac_norms_sum = 0
    # estimate the empirical mean norm of the Jacobians
    with torch.no_grad():
        with tqdm(total=n_tokens_norm_est, desc="Estimating norm") as pbar:
            for item in dataset:
                _, cache = model.run_with_cache(
                    item["text"],
                    stop_at_layer=layer + 1,
                    names_filter=[sae_pair.cfg.hook_name],
                )
                acts = cache[sae_pair.cfg.hook_name][:, 1:]

                if acts.shape[1] > args.context_size:  # Preventing OOM
                    acts = acts[:, : args.context_size]

                jacobian, _ = run_sandwich(sae_pair, mlp_with_grads, acts)
                jac_norms_sum += jac_norms[args.norm](jacobian).sum().item()

                pbar.update(acts.shape[1])
                if pbar.n >= n_tokens_norm_est:
                    break
    mean_jac_norm = jac_norms_sum / pbar.n
    print("Mean Jacobian norm:", mean_jac_norm)

# for the non-averaged histogram
bins = 1500 if args.norm else 1100
min_val = 0
max_val = 1.5 if args.norm else 1.1
bin_edges = torch.linspace(min_val, max_val, bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
hist = torch.zeros(bins, device=sae_pair.cfg.device)

# for the averaged histogram
summed_abs_jacobians = torch.zeros(
    sae_pair.cfg.d_sae, sae_pair.cfg.d_sae, device=sae_pair.device
)

with torch.no_grad():
    with tqdm(total=n_tokens) as pbar:
        for item in dataset:
            # Todo batching (if I have the time)
            _, cache = model.run_with_cache(
                item["text"],
                stop_at_layer=layer + 1,
                names_filter=[sae_pair.cfg.hook_name],
            )
            acts = cache[sae_pair.cfg.hook_name][:, 1:]

            if acts.shape[1] > args.context_size:  # Preventing OOM
                acts = acts[:, : args.context_size]

            jacobian, acts_dict = run_sandwich(sae_pair, mlp_with_grads, acts)

            if args.norm:
                jacobian /= mean_jac_norm

            hist += torch.histc(jacobian.abs().flatten(), bins=bins, min=min_val, max=max_val)

            topk1 = acts_dict["topk_indices1"].unsqueeze(-1)
            topk2 = acts_dict["topk_indices2"].unsqueeze(-2)
            summed_abs_jacobians[topk2, topk1] += jacobian.abs().sum(dim=(0, 1))

            pbar.update(acts.shape[1])
            if pbar.n >= n_tokens:
                break

mean_abs_jacobian_flat = (summed_abs_jacobians / pbar.n).flatten()
mean_jac_hist = torch.zeros(bins, device=sae_pair.cfg.device)
batch_size = 1_048_576
for i in range(math.ceil(mean_abs_jacobian_flat.shape[0] / batch_size)):
    if (i + 1) * batch_size < mean_abs_jacobian_flat.shape[0]:
        sliced_jac = mean_abs_jacobian_flat[i * batch_size : (i + 1) * batch_size]
    else:
        sliced_jac = mean_abs_jacobian_flat[i * batch_size :]

    mean_jac_hist += torch.histc(sliced_jac, bins=bins, min=min_val, max=max_val)

tensors_dict = {
    "hist": hist,
    "mean_jac_hist": mean_jac_hist,
    "bin_edges": bin_edges,
    "bin_centers": bin_centers,
}
metadata = {
    "context_size": str(args.context_size),
    "path": args.path,
    "tokens": str(n_tokens),
}

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(script_dir, args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, f"{args.path.split("/")[-1]}{f'_normed{args.norm}' if args.norm else ''}.safetensor")
save_file(tensors_dict, output_path, metadata=metadata)
