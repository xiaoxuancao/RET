import pandas as pd
import numpy as np
import json
import os
import time


import dask.dataframe as dd

# def get_dataset_name(M, efC, efS, s, ds_name, k, logint):
#     return f"{os.environ['DARTH_ROOT']}/et_training_data/early-stop-training/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{s}_li{logint}.txt"

TRAINING_DATA_DIRECTORY="/data/darth"
def get_dataset_name(M, efC, efS, num_queries, ds_name, k, type, logint):
    return f"{TRAINING_DATA_DIRECTORY}/{ds_name}_{type}training_k{k}_M{M}_efC{efC}_efS{efS}_qs{num_queries}_li{logint}.txt"


def extract_data_stats(data):    
    results = {}
    
    target_recalls = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.00]
    grouped_by_query = data.groupby("qid")
    
    results["correlations_to_target"] = {}
    correlations_to_target = {}
    for col in data.columns:
        if col != "r":
            correlations_to_target[col] = data["r"].corr(data[col])
    
    correlations_to_target = {k: v for k, v in sorted(correlations_to_target.items(), key=lambda item: abs(item[1]), reverse=True)}
    for col, corr in correlations_to_target.items():
        #print(f"        {col}: {corr:.4f}")
        results["correlations_to_target"][col] = corr
    
    results["distance_calcs_to_reach_target_recall"] = {}
    for target_recall in target_recalls:
        distance_calcs_to_reach_target_recall = []
        for idx, group in grouped_by_query:
            group = group[group["r"] >= target_recall]
            if group.shape[0] > 0:
                row_of_first_reached_recall = group.iloc[0]
                distance_calcs_to_reach_target_recall.append(row_of_first_reached_recall["dists"])
        
        mean_dists = np.mean(distance_calcs_to_reach_target_recall)
        max_dists = np.max(distance_calcs_to_reach_target_recall)
        min_dists = np.min(distance_calcs_to_reach_target_recall)
        results["distance_calcs_to_reach_target_recall"][target_recall] = {
            "avg": mean_dists,
            "max": max_dists,
            "min": min_dists
        }
    
    return results

dataset_params = {
        "deep": {
            "ds_name": "deep",
            "M": 32,
            "efC": 500,
            "efS": 500,
            "k": 100,
            "type": ""
        }, 
        "gist": {
            "ds_name": "gist",
            "M": 32,
            "efC": 500,
            "efS": 2000,
            "k": 100,
            "type": ""
        },       
        "glove": {
            "ds_name": "glove",
            "M": 32,
            "efC": 500,
            "efS": 2000,
            "k": 100, 
           "type": ""  
        },
         "laion": {
            "ds_name": "laion",
            "M": 32,
            "efC": 500,
            "efS": 2500,
            "k": 100, 
            "type": ""  
        },
        "t2i": {
            "ds_name": "t2i",
            "M": 32,
            "efC": 500,
            "efS": 500,
            "k": 100,
            "type": ""
        },        
        "laion": {
            "ds_name": "laion",
            "M": 32,
            "efC": 500,
            "efS": 3000,
            "k": 100, 
            "type": "ood-"  
        },
        "t2i": {
            "ds_name": "t2i",
            "M": 32,
            "efC": 500,
            "efS": 2500,
            "k": 100,
            "type": "ood-"
        }
    }

index_metric_feats = ["step", "dists", "inserts"]
neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
neighbor_stats_feats = ["avg_dist", "variance", "percentile_25", "percentile_50", "percentile_75"]
all_feats = index_metric_feats + neighbor_distances_feats + neighbor_stats_feats
columns_to_load = ["qid", "elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]


all_k_values = [10, 50] # dummy k value, as we use different k values for different datasets

stats_file = "../train_log/data_stats.json"
os.makedirs(os.path.dirname(stats_file), exist_ok=True)

for ds_name in dataset_params.keys():
    dataset_stats = {}
    
    M = dataset_params[ds_name]["M"]
    efC = dataset_params[ds_name]["efC"]
    efS = dataset_params[ds_name]["efS"]
    type = dataset_params[ds_name]["type"]
    ds_name = dataset_params[ds_name]["ds_name"]
    for k in all_k_values:
        # k = dataset_params[ds_name]["k"] # actual k value
        print(f"{ds_name} | k={k}")
        li = 2
        dask_df = dd.read_csv(get_dataset_name(M, efC, efS, 10000, ds_name, k, type, li), usecols=columns_to_load) #type: ignore
        data_all = dask_df.compute()
        start_time = time.time()
        #dataset_stats[k] = extract_data_stats(data_all)
        stats_result = extract_data_stats(data_all)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Time taken to extract stats for "
            f"{ds_name} | k={k}: {elapsed_time:.6f} seconds"
        )
        stats_result["extract_time_seconds"] = elapsed_time
        dataset_stats[k] = stats_result
    
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            existing_stats = json.load(f)
    else:
        existing_stats = {}
    existing_stats[ds_name] = dataset_stats
    with open(stats_file, "w") as f:
        json.dump(existing_stats, f, indent=4)

    print(f"Saved stats for {ds_name} to {stats_file}")           
    