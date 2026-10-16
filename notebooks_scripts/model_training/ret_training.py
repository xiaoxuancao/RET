import os
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
import csv
from datetime import datetime
from pprint import pprint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




def compute_P99(y_true, y_pred):
    y_diff = np.abs(y_true - y_pred)
    return np.percentile(y_diff, 99)

def compute_P1(y_true, y_pred):
    y_diff = np.abs(y_true - y_pred)
    return np.percentile(y_diff, 1)

def load_data(data_path, columns_to_load, training_num=None):
    if training_num is not None:
        all_queries_dask = dd.read_csv(data_path, usecols=["qid"] + columns_to_load)
        data_all = all_queries_dask.compute()
        unique_qids = data_all['qid'].unique()[:training_num]
        data_all = data_all[data_all['qid'].isin(unique_qids)]
    else:
        all_queries_dask = dd.read_csv(data_path, usecols=columns_to_load)
        data_all = all_queries_dask.compute()
    
    print(f"    Queries Data Shape: {data_all.shape} | Unique queries: {data_all['qid'].nunique()}")
    return data_all


def train_predictors(model_file, train_data, n_estimators, feature_classes, model_conf):
    results = {}
            
    y_all = train_data["r"]
        
    for selected_features in feature_classes.keys():
        feats = feature_classes[selected_features]
                                    
        X_all = train_data[feats]
        X_train, y_train = X_all, y_all
            
        model = model_conf["lightgbm"]
        model_train_time_start = time.time()
        model.fit(X_train, y_train)
        model_train_time = time.time() - model_train_time_start  
            
        model_params = model.get_params()
        learning_rate = model_params["learning_rate"]
            
        feature_importances = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_})
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
        
        model.booster_.save_model(model_file)
        print(f"        Model Train Time: {model_train_time:.2f} | learning rate: {learning_rate}")

        results[selected_features] = {
            "training_time": model_train_time,
            "training_data_size": X_train.shape[0],
            "learning_rate": learning_rate,
            "feature_importances": feature_importances.to_dict(orient="records"),
            "n_estimators": n_estimators,
        }
        
        # print(f"results[{selected_features}]:")
        # pprint(results[selected_features])
                      
    print(f"Finished model training")
    
    return model, results

def save_feature_analysis(csv_file, feature_corr, feature_importances, ds_name, training_num, M, efC, efS, k, type):
   
    corr_df = feature_corr.reset_index()
    corr_df.columns = ['Feature', 'Correlation_with_r']
    corr_df['ds_name'] = ds_name
    corr_df['training_num'] = training_num
    corr_df['M'] = M
    corr_df['efC'] = efC
    corr_df['efS'] = efS
    corr_df['k'] = k
    corr_df['type'] = type
    

    imp_df = feature_importances.copy()
    imp_df['ds_name'] = ds_name
    imp_df['training_num'] = training_num
    imp_df['M'] = M
    imp_df['efC'] = efC
    imp_df['efS'] = efS
    imp_df['k'] = k
    imp_df['type'] = type
    
    return corr_df, imp_df

def validate_predictors(model, validating_data, feature_classes):
    results = {}

    validation_y_true = validating_data["r"]
    
    for selected_features in feature_classes.keys():
        feats = feature_classes[selected_features]
        validation_X = validating_data[feats]
        validation_y_pred = model.predict(validation_X)
        mse = mean_squared_error(validation_y_true, validation_y_pred)
        mae = mean_absolute_error(validation_y_true, validation_y_pred)
        r2 = r2_score(validation_y_true, validation_y_pred)
        p99 = compute_P99(validation_y_true, validation_y_pred)
        p1 = compute_P1(validation_y_true, validation_y_pred)
    
        average_relative_error = np.mean(np.abs(validation_y_true - validation_y_pred) / validation_y_true)
        maximum_relative_error = np.max(np.abs(validation_y_true - validation_y_pred) / validation_y_true)
    
        results[selected_features] = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "p99": p99,
            "p1": p1,
            "average_relative_error": average_relative_error,
            "maximum_relative_error": maximum_relative_error
        }
        # print(f"results[{selected_features}]:\n{results[selected_features]}")
    return results


def main():
    SEED = 42
    n_estimators = 100

    all_corr_data = []
    all_imp_data = []

    feature_cols = [
        "step",
        "dists",
        "inserts",

        "steps_diff",
        "dists_diff",

        "candidate_dist_diff",
        "no_insert_max_length",
        "no_insert_count",

        "kth_nn_start",
        "first_nn_start",

        "first_nn_kth",
        "first_nn_9kth",
        "first_nn_8kth",
        "first_nn_7kth",

        "kth_nn_9kth",
        "kth_nn_8kth",
        "kth_nn_7kth",

        "lid",
    ]

    columns_to_load = [
        "qid",
        "elaps_ms",
        "feats_collect_time_ms",
        "r"
    ] + feature_cols
   
    dataset_params = {
            # "deep": {
            #     "ds_name": "deep",
            #     "M": 32,
            #     "efC": 500,
            #     "efS": 500,
            #     "type": ""
            # },
            # "gist": {
            #     "ds_name": "gist",
            #     "M": 32,
            #     "efC": 500,
            #     "efS": 2000,
            #     "type": ""
            # },          
            "glove": {
                "ds_name": "glove",
                "M": 32,
                "efC": 500,
                "efS": 2000,
                "type": ""  
            },        
            "laion": {
                "ds_name": "laion",
                "M": 32,
                "efC": 500,
                "efS": 2500,
                "type": ""  
            },
            "t2i": {
                "ds_name": "t2i",
                "M": 32,
                "efC": 500,
                "efS": 500,
                "type": ""
            },
            "laion-ood": {
                "ds_name": "laion",
                "M": 32,
                "efC": 500,
                "efS": 3000,
                "type": "ood-"  
            },
            "t2i-ood": {
                "ds_name": "t2i",
                "M": 32,
                "efC": 500,
                "efS": 2500,
                "type": "ood-"
            }
        }

    # training_num_list = [100, 500, 1000, 5000, None]
    training_num_list = [None]

    k_num_list = [10, 50]

    results_file = f"../train_log/retmodel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    all_results = []

    all_datasets = dataset_params.keys()
    for k in k_num_list:
        for ds_name_key in all_datasets:
            M = dataset_params[ds_name_key]["M"]
            efC = dataset_params[ds_name_key]["efC"]
            efS = dataset_params[ds_name_key]["efS"]
            type = dataset_params[ds_name_key]["type"]
            ds_name = dataset_params[ds_name_key]["ds_name"]
            
            training_path = f"/data/ret/{ds_name}_{type}training_k{k}_M{M}_efC{efC}_efS{efS}_qs10000.txt"
            validation_path = f"/data/ret/{ds_name}_{type}validation_k{k}_M{M}_efC{efC}_efS{efS}_qs1000.txt"

            for training_num in training_num_list: 
                print(f"\n===== dataset: {ds_name} | M: {M} | efC: {efC} | efS: {efS} | k: {k} | type: {type} | training_num: {training_num} =====")
                start = time.time()
                train_data = load_data(training_path, columns_to_load, training_num=training_num)
                dta_load_time = time.time() - start
               
                validate_data = load_data(validation_path, columns_to_load)

                feature_classes = {
                    "all_features": feature_cols,
                }

                all_feats = feature_cols
                corr = train_data[all_feats + ["r"]].corr(method="spearman")["r"].sort_values(ascending=False)
                corr_df = corr.reset_index()
                corr_df.columns = ['Feature', 'Correlation']
                corr_df['ds_name'] = ds_name
                corr_df['training_num'] = training_num
                all_corr_data.append(corr_df)

                # print(corr)

                model_conf = {
                    "lightgbm": lgb.LGBMRegressor(objective='regression', random_state=SEED, n_estimators=n_estimators, verbose = -1),
                }
                
                os.makedirs(f"../../model/ret/", exist_ok=True)
                model_file = f"../../model/ret/{ds_name}_M{M}_efC{efC}_efS{efS}_k{k}_nestim{n_estimators}_tn{training_num}_32all.txt"
                start = time.time()
                model, train_results = train_predictors(model_file, train_data, n_estimators, feature_classes, model_conf)
                model_train_time = time.time() - start
                
                imp_df = train_results["all_features"]["feature_importances"]
                if isinstance(imp_df, list):
                    imp_df = pd.DataFrame(imp_df)
                else:
                    imp_df = imp_df.copy()
                imp_df['ds_name'] = ds_name
                imp_df['training_num'] = training_num
                all_imp_data.append(imp_df)
                validate_results = validate_predictors(model, validate_data, feature_classes)

                all_results.append({
                    "ds_name": ds_name,
                    "M": M,
                    "efC": efC,
                    "efS": efS,
                    "k": k,
                    "type": type,
                    "training_num": training_num,
                    "dta_load_time": dta_load_time,
                    "model_train_time": model_train_time,
                    "total_time": dta_load_time + model_train_time,
                    "mse": validate_results["all_features"]["mse"],
                    "mae": validate_results["all_features"]["mae"],
                    "r2": validate_results["all_features"]["r2"],
                    "p99": validate_results["all_features"]["p99"],
                    "p1": validate_results["all_features"]["p1"],
                    "average_relative_error": validate_results["all_features"]["average_relative_error"],
                    "maximum_relative_error": validate_results["all_features"]["maximum_relative_error"],
                })

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = [
            "ds_name", "M", "efC", "efS", "k", "type", "training_num",
            "dta_load_time", "model_train_time", "total_time",
            "mse", "mae", "r2", "p99", "p1",
            "average_relative_error", "maximum_relative_error"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    # if all_corr_data:
    #         corr_combined = pd.concat(all_corr_data, ignore_index=True)
    #         corr_combined.to_csv("train_log/feature_correlation_all.csv", index=False)
    # if all_imp_data:
    #         imp_combined = pd.concat(all_imp_data, ignore_index=True)
    #         imp_combined.to_csv("train_log/feature_importance_all.csv", index=False)

if __name__ == "__main__":
    main()