# Stage 3 of autointerp: you already have the results (from autointerp.py), now check how many of them are positive
# (Doesn't require a GPU)

import itertools
import os
import json

for model_name, experiment_name in itertools.product(["jsaes", "traditional"], ["detection", "fuzz"]):
    dir_name = f"results/autointerp/{model_name}/scores/{experiment_name}"
    all_data: list[bool] = {}
    for module_name in os.listdir(dir_name):
        if not os.path.isdir(f"{dir_name}/{module_name}"):
            continue

        all_data[module_name] = []

        for file_name in os.listdir(f"{dir_name}/{module_name}"):
            with open(f"{dir_name}/{module_name}/{file_name}") as f:
                for result_dict in json.load(f):
                    all_data[module_name].append(result_dict["correct"])

    with open(f"results/autointerp/{model_name}/scores/{experiment_name}_results.json", "w") as f:
        json.dump(all_data, f)
