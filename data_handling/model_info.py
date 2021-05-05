import json
import numpy as np

def load_model_info(dataset_path, oiu, verbose=1):
    model_info = {}
    with open(f'{dataset_path}/models_eval/models_info.json') as f:
        jsondata = json.load(f)

        key = str(oiu)
        assert(key in jsondata)

        model_info["diameter"] = jsondata[key]["diameter"]
        model_info["mins"] = np.array([jsondata[key]["min_x"],jsondata[key]["min_y"],jsondata[key]["min_z"]])
        model_info["maxs"] = np.array([jsondata[key]["size_x"],jsondata[key]["size_y"],jsondata[key]["size_z"]]) +  model_info["mins"]

        model_info["symmetries_discrete"] = [np.array(_).reshape((4,4)) for _ in jsondata[key]["symmetries_discrete"]] if "symmetries_discrete" in jsondata[key] else []
        model_info["symmetries_continuous"] = "symmetries_continuous" in jsondata[key]

    if verbose > 0:
        print (f'model_info for object {oiu}:')
        for k in model_info:
            print (k, ':', model_info[k])
            
    return model_info