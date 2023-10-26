import json

import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_gglf_configs(gglf_configs: dict, file_name: str):
    json_object = json.dumps(gglf_configs, indent=4, cls=NpEncoder)

    with open(file_name, "w") as outfile:
        outfile.write(json_object)


def load_gglf_configs(file_name: str):
    with open(file_name, "r") as infile:
        json_object = json.load(infile)

    return json_object
