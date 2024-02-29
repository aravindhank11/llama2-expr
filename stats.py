# Sample command: python3 new_stats.py --mode mps-uncap --load 0.1 --result_dir dir1 /tmp/111003.pkl /tmp/111004.pkl /tmp/111005.pkl

import argparse
import os
import sys
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd


def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def populate_stats(id, array, metrics):
    # Calculate percentiles
    metrics[f"{id}_p0"].append(np.min(array) * 1000)
    metrics[f"{id}_p50"].append(np.percentile(array, 50) * 1000)
    metrics[f"{id}_p90"].append(np.percentile(array, 90) * 1000)
    metrics[f"{id}_p99"].append(np.percentile(array, 99) * 1000)
    metrics[f"{id}_p100"].append(np.max(array) * 1000)


def create_dir(directory):
    try:
        os.makedirs(directory)
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--load", type=float, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument(
        "pickle_files",
        metavar="<pkl_file>",
        nargs="+",
        help="List of model and details"
    )
    opt = parser.parse_args()

    create_dir(opt.result_dir)

    ctr = 1
    models = []
    metrics = defaultdict(list)
    for pickle_file in opt.pickle_files:
        # Validate file paths
        if not os.path.isfile(pickle_file):
            print(f"File '{pickle_file}' does not exist.")
            sys.exit(1)

        # Load arrays from pickle files
        model, tput, total_times, queued_times = load_pickle_file(pickle_file)
        populate_stats("total", total_times, metrics)
        populate_stats("queued", queued_times, metrics)
        metrics["tput"].append(tput)
        models.append(f"{ctr}_{model}")
        ctr += 1

    # Create a DataFrame for each metric type
    for metric_type, metrics_list in metrics.items():
        df_data = {
            model_name: metrics_list[i] for i, model_name in enumerate(models)
        }
        df_data["mode"] = opt.mode
        df_data["load"] = opt.load
        df = pd.DataFrame(df_data, index=[metric_type])
        cols = (["mode", "load"] +
                [col for col in df.columns if col != "mode" and col != "load"])
        df = df[cols]


        csv_file = os.path.join(opt.result_dir, f"{metric_type}.csv")
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)
