import argparse
import sys
import os

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from jacobian_saes.utils import load_pretrained, run_sandwich


# parse args
parser = argparse.ArgumentParser(
    description="Run the causal dependence ablation experiment"
)
parser.add_argument(
    "--output-dir",
    "-o",
    type=str,
    default="results/ablation",
    help="Output directory",
)
parser.add_argument(
    "--path",
    "-p",
    type=str,
    default="lucyfarnik/jsaes_pythia70m1/sae_pythia-70m-deduped_blocks.3.ln2.hook_normalized_16384:v0",
    help="Wandb path to the SAE pair artifact",
)
parser.add_argument(
    "--samples",
    "-s",
    type=float,
    default=10_000_000,
    help="Total number of samples to collect",
)
args = parser.parse_args()

n_samples = int(args.samples)

sae_pair, model, mlp_with_grads, layer = load_pretrained(args.path)
k = sae_pair.cfg.activation_fn_kwargs["k"]

dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

# each sample is [value in Jacobian, diff in output feature when ablating input feature]
results = torch.zeros(n_samples, 2)
with torch.no_grad():
    with tqdm(total=n_samples) as pbar:
        for item in dataset:
            _, cache = model.run_with_cache(
                item["text"],
                stop_at_layer=layer + 1,
                names_filter=[sae_pair.cfg.hook_name],
            )
            acts = cache[sae_pair.cfg.hook_name][:, 1:]

            for act in acts[0, 1:]:
                jacobian, acts_dict = run_sandwich(sae_pair, mlp_with_grads, act)
                sae_acts2 = acts_dict["sae_acts2"]
                topk_indices1 = acts_dict["topk_indices1"]
                topk_indices2 = acts_dict["topk_indices2"]

                for out_idx, in_idx in zip(range(k), range(k)):
                    results[pbar.n, 0] = jacobian[out_idx, in_idx]

                    in_idx_in_d_sae = topk_indices1[in_idx]
                    out_idx_in_d_sae = topk_indices2[out_idx]

                    act_abl = act - sae_pair.get_W_dec(False)[in_idx_in_d_sae]
                    mlp_out_abl, _ = mlp_with_grads(act_abl)
                    sae_acts2_abl = sae_pair.encode(mlp_out_abl, True)

                    results[pbar.n, 1] = sae_acts2_abl[out_idx_in_d_sae] - sae_acts2[out_idx_in_d_sae]

                    pbar.update(1)
                    if pbar.n >= n_samples:
                        break
                if pbar.n >= n_samples:
                    break
            if pbar.n >= n_samples:
                break


tensors_dict = {
    "results": results,
}
metadata = {
    "path": args.path,
    "samples": str(n_samples),
}

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(script_dir, args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, f"{args.path.split("/")[-1]}.safetensor")
save_file(tensors_dict, output_path, metadata=metadata)
