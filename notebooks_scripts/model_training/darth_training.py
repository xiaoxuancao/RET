import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time
import json

from concurrent.futures import ThreadPoolExecutor, as_completed

import dask.dataframe as dd

TRAINING_DATA_DIRECTORY="/data/darth"

def _series_memory_bytes(series: pd.Series) -> int:
    if series.empty:
        return 0
    return len(series) * 4 # assuming float32 or int32

def _dataframe_memory_bytes(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    row_count = len(df)
    column_bytes = sum(4 for _ in df.dtypes) # assuming float32 or int32
    return row_count * column_bytes

def compute_memory_footprint(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[int, int]:
    return _dataframe_memory_bytes(X_train), _series_memory_bytes(y_train)

def get_dataset_name(M, efC, efS, num_queries, ds_name, k, type, logint):
    return f"{TRAINING_DATA_DIRECTORY}/{ds_name}_{type}training_k{k}_M{M}_efC{efC}_efS{efS}_qs{num_queries}_li{logint}.txt"

def main():
    SEED = 42
    n_estimators = 100

    index_metric_feats = ["step", "dists", "inserts"]
    neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
    neighbor_stats_feats = ["avg_dist", "variance", "percentile_25", "percentile_50", "percentile_75"]
    all_feats = index_metric_feats + neighbor_distances_feats + neighbor_stats_feats

    columns_to_load = ["qid", "elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]
        
    model_conf = {
        "lightgbm": lgb.LGBMRegressor(objective='regression', random_state=SEED, n_estimators=n_estimators, verbose = -1),
    }

    feature_classes = {
        #"index_metric_feats": index_metric_feats,
        #"neighbor_distances_feats": neighbor_distances_feats,
        #"neighbor_stats_feats": neighbor_stats_feats,
        #"index_metrics_and_neighbor_distances": index_metric_feats + neighbor_distances_feats,
        #"index_metrics_and_neighbor_stats": index_metric_feats + neighbor_stats_feats,
        #"neighbor_distances_and_neighbor_stats": neighbor_distances_feats + neighbor_stats_feats,
        "all_feats": all_feats,
    }
    
    dataset_params = {
        "deep": {
            "M": 32,
            "efC": 500,
            "efS": 500,
            "k": 100,
            "type": ""
        },    
        "gist": {
            "M": 32,
            "efC": 500,
            "efS": 2000,
            "k": 100, 
            "type": ""  
        },     
        "glove": {
            "M": 32,
            "efC": 500,
            "efS": 2000,
            "k": 100, 
            "type": ""  
        },        
        "laion": {
            "M": 32,
            "efC": 500,
            "efS": 2500,
            "k": 100,
            "type": ""   
        },
        "t2i": {
            "M": 32,
            "efC": 500,
            "efS": 500,
            "k": 100,
            "type": ""
        },
        "laion-ood": {
            "M": 32,
            "efC": 500,
            "efS": 3000,
            "k": 100,
            "type": "ood-"   
        },
        "t2i-ood": {
            "M": 32,
            "efC": 500,
            "efS": 2500,
            "k": 100,
            "type": "ood-"
        }
    }

    training_queries_num = [10000]
    all_datasets = dataset_params.keys()
    all_li = [2]

    for k in [10, 50]:    
        for ds_name in all_datasets:
            M = dataset_params[ds_name]["M"]
            efC = dataset_params[ds_name]["efC"]
            efS = dataset_params[ds_name]["efS"]
            # k = dataset_params[ds_name]["k"]
            type = dataset_params[ds_name]["type"]

            results = train_predictors(
                ds_name, training_queries_num, [k], type, all_li,
                M, efC, efS, n_estimators, columns_to_load, feature_classes, model_conf)

            with open(f"../train_log/training_results_{ds_name}_{type}_u10K_{all_li[0]}_efS{efS}.json", "w") as f:
                json.dump(results, f, indent=4)
    
def train_predictors(ds_name, training_queries_num, all_k_values, type, all_li, M, efC, efS, n_estimators, columns_to_load, feature_classes, model_conf):
    results = {}
    
    max_query_size = max(training_queries_num)
    
    print(f"Starting training for {ds_name}")
        
    for k in all_k_values:
        results[k] = {}
        li = 2
        
        data_path = get_dataset_name(M, efC, efS, max_query_size, ds_name, k, type, li)
        
        print(f"{ds_name} | k={k} | Path: {data_path}")
        
        start = time.time()
        all_queries_dask = dd.read_csv(data_path, usecols=columns_to_load) #type: ignore
        all_queries_data = all_queries_dask.compute()
        dta_load_time = time.time() - start
      
        
        for s in training_queries_num:
            results[k][s] = {}
            query_data_all = all_queries_data[all_queries_data["qid"] < s]
                
            for li in all_li:
                results[k][s][li] = {}
                    
                data_all = query_data_all[query_data_all["dists"] % li == 0]
                    
                print(f"    {s} Queries Data Shape: {data_all.shape} |  Unique queries: {data_all['qid'].nunique()} | Li: {li}")
                        
                y_all = data_all["r"]
                    
                for selected_features in feature_classes.keys():
                    feats = feature_classes[selected_features]
                                                
                    X_all = data_all[feats]
                    X_train, y_train = X_all, y_all
                    x_memory_bytes, y_memory_bytes = compute_memory_footprint(X_train, y_train)
                    model = model_conf["lightgbm"]
                    model_train_time_start = time.time()
                    model.fit(X_train, y_train)
                    model_train_time = time.time() - model_train_time_start                      
                    model_params = model.get_params()
                    learning_rate = model_params["learning_rate"]
                        
                    feature_importances = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_})
                    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
                    
                    # create directory if not exists
                    os.makedirs(f"../../model/darth/", exist_ok=True)
                    model_file = f"../../model/darth/{ds_name}_M{M}_efC{efC}_efS{efS}_s{s}_k{k}_nestim{n_estimators}_li{li}_{selected_features}.txt"
                    model.booster_.save_model(model_file)                    
                    model_memory_bytes = os.path.getsize(model_file)
                    print(f"        Model Train Time: {model_train_time:.6f} | learning rate: {learning_rate} | Saved to: {model_file}")
                    print(f" the total time: {dta_load_time + model_train_time:.6f} seconds")
                    results[k][s][li][selected_features] = {
                        "dta_load_time": dta_load_time,
                        "training_time": model_train_time,
                        "the_total_time": dta_load_time + model_train_time,
                        "training_data_size": X_train.shape[0],
                        "learning_rate": learning_rate,
                        "feature_importances": feature_importances.to_dict(orient="records"),
                        "n_estimators": n_estimators,
                        "X_memory_bytes": x_memory_bytes,
                        "y_memory_bytes": y_memory_bytes,
                        "model_memory_bytes": model_memory_bytes,
                    }
                    
                    print(f"results[{k}][{s}][{li}][{selected_features}]:\n{results[k][s][li][selected_features]}")
                    
            print()
        print("\n\n")    
    print(f"Finished training for {ds_name}")
    
    return results
    

if __name__ == "__main__":
    main()