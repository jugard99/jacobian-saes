# Stage 3 of autointerp: you already have the results (from autointerp.py), now collect them into a single file
# (Doesn't require a GPU)

import itertools
import numpy as np
import os
import json
from tqdm import tqdm

# for model_name, experiment_name in itertools.product(["jsaes", "traditional"], ["detection", "fuzz"]):
experiment_name = "fuzz"
for model_name in ["jsaes", "traditional"]:
    dir_name = f"results/autointerp/{model_name}/scores/{experiment_name}"
    all_data: list[bool] = {}
    for module_name in tqdm(os.listdir(dir_name), desc=f"{model_name} {experiment_name}"):
        if not os.path.isdir(f"{dir_name}/{module_name}"):
            continue

        all_bools = []
        #TODO parallelize
        for file_name in os.listdir(f"{dir_name}/{module_name}"):
            with open(f"{dir_name}/{module_name}/{file_name}") as f:
                for result_dict in json.load(f):
                    all_bools.append(result_dict["correct"])
        if len(all_bools) == 0:
            continue
        all_bools = np.array(all_bools)
        
        num_true = np.sum(all_bools).tolist()
        all_data[module_name] = {
            "mean": num_true / len(all_bools),
            "num_true": num_true,
            "num_false": len(all_bools) - num_true,
        }

    with open(f"results/autointerp/{model_name}_{experiment_name}_results.json", "w") as f:
        json.dump(all_data, f)
