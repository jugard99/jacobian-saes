import argparse
from datasets import load_dataset
import json
import torch
import os
import sys
from tqdm import tqdm
from transformer_lens.utils import get_act_name

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from jacobian_saes.utils import load_pretrained

parser = argparse.ArgumentParser(
    description="Check how linear f_s is for a give pair of SAEs"
)
parser.add_argument(
    "--output-dir",
    "-o",
    type=str,
    default="results/linear",
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
    "--samples",
    "-s",
    type=float,
    default=1_000_000,
    help="Total number of samples to check; 1 sample = one set of (s_x, i, j)",
)
args = parser.parse_args()

n_samples = int(args.samples)

sae_pair, model, mlp_with_grads, layer = load_pretrained(args.path)
k = sae_pair.cfg.activation_fn_kwargs["k"]
mlp_out_hook_name = get_act_name("mlp_out", sae_pair.cfg.hook_layer)

dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)


def is_linear(xs: torch.Tensor, ys: torch.Tensor, tolerance=1e-2) -> bool:
    # Fit a linear model to the outside region (y = slope * x + intercept)
    A = torch.stack([xs, torch.ones_like(xs)], dim=1)  # [x, 1]
    lstsq = torch.linalg.lstsq(A.cpu(), ys.unsqueeze(1).cpu())  # Linear regression
    slope = lstsq.solution[0].item()
    intercept = lstsq.solution[1].item()

    # Check if residuals are within tolerance
    residuals = ys - (slope * xs + intercept)
    return torch.all(torch.abs(residuals) < tolerance).item()


def is_jump_relu(xs: torch.Tensor, ys: torch.Tensor, tolerance=1e-2):
    # Check for zero region
    zero_region = (ys == 0)

    if not torch.any(zero_region):
        return False

    # Find the bounds of the zero region
    zero_indices = torch.where(zero_region)[0]
    zero_start, zero_end = xs[zero_indices[0]], xs[zero_indices[-1]]

    # Make sure the zero region is continuous
    if not (ys[zero_indices[0]:zero_indices[-1]] == 0).all():
        return False

    # Make sure it includes the start or the finish
    if not ys[0] == 0 and not ys[-1] == 0:
        return False

    # Check linearity outside the zero region
    linear_region_mask = (xs > zero_end) | (xs < zero_start)
    linear_x = xs[linear_region_mask]
    linear_y = ys[linear_region_mask]

    zero_start, zero_end = zero_start.item(), zero_end.item()

    if linear_x.numel() < 2:
        return False

    return is_linear(linear_x, linear_y, tolerance)


is_linear_count = 0
is_jump_relu_count = 0
is_neither_count = 0
with torch.no_grad():
    with tqdm(total=n_samples) as pbar:
        for item in dataset:
            # cache the activations
            _, cache = model.run_with_cache(
                item["text"],
                stop_at_layer=layer + 1,
                names_filter=[sae_pair.cfg.hook_name, mlp_out_hook_name],
            )
            acts = cache[sae_pair.cfg.hook_name][0, 1:]
            mlp_out_acts = cache[mlp_out_hook_name][0, 1:]

            # for each sequence position
            for act, mlp_out in zip(acts, mlp_out_acts):
                # get SAE activations and topk indices
                sae_acts1, topk_indices1 = sae_pair.encode(
                    act, False, return_topk_indices=True)
                sae_acts2, topk_indices2 = sae_pair.encode(
                    mlp_out, True, return_topk_indices=True)

                # pick a few random pairs of indices
                for _ in range(k):  # doesn't have to be k, this can be any number
                    out_idx, in_idx = torch.randint(0, k, (2,))

                    in_idx_in_d_sae = topk_indices1[in_idx]
                    out_idx_in_d_sae = topk_indices2[out_idx]

                    # get the direction and strength of the in feature
                    in_feature_dir = sae_pair.get_W_dec(False)[in_idx_in_d_sae]
                    in_feature_strength = sae_acts1[in_idx_in_d_sae]

                    # create a range of activations that vary by the strength of the feature
                    max_upstream = max(5, in_feature_strength+1)
                    upstream_acts = torch.linspace(0, max_upstream, 100,
                                                   device=act.device).reshape(-1, 1)
                    act_abl = act - in_feature_strength * in_feature_dir
                    act_range = upstream_acts * in_feature_dir + act_abl.unsqueeze(0)
                    mlp_out_range, _ = mlp_with_grads(act_range)
                    sae_acts2_range = sae_pair.encode(mlp_out_range, True)
                    downstream_acts = sae_acts2_range[:, out_idx_in_d_sae]

                    # check if it's linear or JumpReLU
                    upstream_acts_flat = upstream_acts.flatten()
                    if is_linear(upstream_acts_flat, downstream_acts):
                        is_linear_count += 1
                    elif is_jump_relu(upstream_acts_flat, downstream_acts):
                        is_jump_relu_count += 1
                    else:
                        is_neither_count += 1

                    pbar.update(1)
                    if pbar.n >= n_samples:
                        break
                if pbar.n >= n_samples:
                    break
            if pbar.n >= n_samples:
                break

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(script_dir, args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, f"{args.path.split("/")[-1]}.json")

results_dict = {
    "is_linear_count": is_linear_count,
    "is_jump_relu_count": is_jump_relu_count,
    "is_neither_count": is_neither_count,
    "path": args.path,
}

with open(output_path, "w") as f:
    json.dump(results_dict, f)
