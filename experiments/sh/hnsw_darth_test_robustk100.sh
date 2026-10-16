#!/usr/bin/env bash
# cd ../
# cmake -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -B build -S .
# make -C build -j faiss
# make -C build -j hnsw_test
# cd experiments

echo ""
echo ""
echo "============================="
echo ""
echo ""


mode=early-stop-testing
n_estim=100
train_queries=10000
experiment_times=1

INDEX_DIRECTORY=/data/
DATASET_DIRECTORY=/data/
MODEL_DIRECTORY=../../model

# target_recall_list=(0.7 0.75 0.8 0.85 0.9 0.95 0.99 1)
# target_recall_list=(0.9)


#parameter, ds M efC efS li k sample initial_prediction_interval min_prediction_interval model query_type target_recall
dataset_params=(
    
    # ood
   
    # laion
    "laion      32 500 3000 2 100 1000  2939  588  3000 ood-testing 0.8"
    "laion      32 500 3000 2 100 1000  3800  760  3000 ood-testing 0.85"
    "laion      32 500 3000 2 100 1000  4962  992  3000 ood-testing 0.9"
    "laion      32 500 3000 2 100 1000  6680 1336  3000 ood-testing 0.95"
    "laion      32 500 3000 2 100 1000  8867 1773  3000 ood-testing 0.99"

    # t2i
    "t2i        32 500 2500 2 100 1000  1683  337  2500 ood-testing 0.8"
    "t2i        32 500 2500 2 100 1000  2079  416  2500 ood-testing 0.85"
    "t2i        32 500 2500 2 100 1000  2699  540  2500 ood-testing 0.9"
    "t2i        32 500 2500 2 100 1000  3847  769  2500 ood-testing 0.95"
    "t2i        32 500 2500 2 100 1000  6018 1204  2500 ood-testing 0.99"
#     # robustness
    "laion      32 500 3000 2 100 1000 646  129  2500 ood-testing 0.8"
    "laion      32 500 3000 2 100 1000 802  160  2500 ood-testing 0.85"
    "laion      32 500 3000 2 100 1000 1053 211  2500 ood-testing 0.9"
    "laion      32 500 3000 2 100 1000 1518 304  2500 ood-testing 0.95"
    "laion      32 500 3000 2 100 1000 2424 485  2500 ood-testing 0.99"
    # t2i
   
    "t2i        32 500 2500 2 100 1000 363  73   500 ood-testing 0.8"
    "t2i        32 500 2500 2 100 1000 443  89   500 ood-testing 0.85"
    "t2i        32 500 2500 2 100 1000 563  113  500 ood-testing 0.9"
    "t2i        32 500 2500 2 100 1000 778  156  500 ood-testing 0.95"
    "t2i        32 500 2500 2 100 1000 1284 257  500 ood-testing 0.99"
)
#mkdir -p ${RESULTS_DIRECTORY}/${mode}

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS li k sample initial_prediction_interval min_prediction_interval model query_type target_recall <<< "$dataset_param"

    # for target_recall in "${target_recall_list[@]}"
    # do
    
        echo ""
        echo "--------------- ${ds}, k=${k}, efS=${efS}, model=${model}, query_type=${query_type} ---------------"
        echo "--------------- target_recall=${target_recall} ---------------"
        echo ""


        for time in $(seq 1 $experiment_times)
        do

            ./../../build/hnsw-test/hnsw_test \
            --dataset ${ds} \
            --M ${M} --efConstruction ${efC} --efSearch ${efS} \
            --query-num ${sample} --k $k \
            --mode ${mode} \
            --index-filepath ${INDEX_DIRECTORY}/${ds}/fevcs/${ds}.M${M}.efC${efC}.index \
            --dataset-dir-prefix ${DATASET_DIRECTORY} \
            --target-recall ${target_recall} \
            --initial-prediction-interval ${initial_prediction_interval} \
            --min-prediction-interval ${min_prediction_interval} \
            --query-type ${query_type} \
            --predictor-model-path ${MODEL_DIRECTORY}/darth/${ds}_M${M}_efC${efC}_efS${model}_s${train_queries}_k${k}_nestim${n_estim}_li${li}_all_feats.txt \
            --output ../../logs/darth/${ds}_${query_type}_k${k}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}_t${time}_m${model}.txt 
        done
    # done
    # python result_merger.py \
    #       ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval} \
    #       ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}.txt \
    #       $experiment_times
    echo ""
    echo ""
done