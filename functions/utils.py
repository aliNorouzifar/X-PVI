import json
import pandas as pd

def save_variables(dict):
    try:
        with open("output_files/internal_variables.json", "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}

    for k in dict.keys():
        if isinstance(dict[k], pd.DataFrame):
            data[k] = dict[k].to_json(orient="split")
        else:
            data[k] = dict[k]

    # data = {
    #     "df": df_json,
    #     "masks": masks,
    #     "map_range": map_range,
    #     "peaks": peaks
    # }

    # Save to a JSON file
    with open("output_files/internal_variables.json", "w") as json_file:
        json.dump(data, json_file)

def load_variables():
    try:
        with open("output_files/internal_variables.json", "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        return "No data file found."
    df = pd.read_json(data["df"], orient="split")
    data["df"] = df
    return data