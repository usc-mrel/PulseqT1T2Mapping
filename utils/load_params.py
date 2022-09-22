import json
import os

def load_params(param_filename:str, param_dir: str = "protocols"):
    # Load parameter file
    params = {}

    with open(os.path.join(param_dir,param_filename + ".json"), "r") as fj:
        jsonstr = fj.read()
        params = json.loads(jsonstr)

    return params