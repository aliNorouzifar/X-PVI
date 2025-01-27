import json
import pandas as pd

def save_variables(dict, file):
    file_path = f"output_files/{file}.json"
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}


    for k in dict.keys():
        if isinstance(dict[k], pd.DataFrame):
            data[k] = dict[k].to_json(orient="split")
        else:
            data[k] = dict[k]

    # Save to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(data, json_file,indent=4)

def load_variables(file):
    try:
        with open(f"output_files/{file}.json", "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        return "No data file found."
    df = pd.read_json(data["df"], orient="split")
    data["df"] = df
    return data