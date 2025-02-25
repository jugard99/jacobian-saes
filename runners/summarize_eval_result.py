import argparse
import json
import os
from safetensors.torch import safe_open


parser = argparse.ArgumentParser(description="Summarize the eval runner summaries (so that we don't have to scp the whole thing if we don't need it)")
parser.add_argument(
    "--output-dir",
    "-o",
    type=str,
    default="results/eval_summaries",
    help="Output directory",
)
parser.add_argument(
    "--path",
    "-p",
    type=str,
    default="results/eval/sae_pair_pythia-410m-deduped_layer14_65536_J0.5_k32:v0.safetensor",
    help="Path to the eval results file",
)
args = parser.parse_args()

with safe_open(args.path, framework="pt", device="cpu") as f:
    summary_data = {}
    for k in f.keys():
        val = f.get_tensor(k)
        if len(val.shape) == 0:
            summary_data[k] = val.item()
    for k, v in f.metadata().items():
        try:
            summary_data[k] = float(v)
        except ValueError:
            summary_data[k] = v

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(script_dir, args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(os.path.join(output_dir, os.path.basename(args.path).replace(".safetensor", "_summary.json")), "w") as f:
    json.dump(summary_data, f)
