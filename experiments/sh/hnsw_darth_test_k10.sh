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
#parameter, ds M efC efS li k sample initial_prediction_interval min_prediction_interval model query_type target_recall
dataset_params=(
    # deep
    "deep       32 500 500  2 10 10000 148 29 500  testing 0.8"
    "deep       32 500 500  2 10 10000 228 45 500  testing 0.85"
    "deep       32 500 500  2 10 10000 228 45 500  testing 0.9"
    "deep       32 500 500  2 10 10000 481 96 500  testing 0.95"
    "deep       32 500 500  2 10 10000 481 96 500  testing 0.99"

    # gist
    "gist       32 500 2000 2 10 1000 867 173 2000 testing 0.8"
    "gist       32 500 2000 2 10 1000 1304 260 2000 testing 0.85"
    "gist       32 500 2000 2 10 1000 1304 260 2000 testing 0.9"
    "gist       32 500 2000 2 10 1000 2346 469 2000 testing 0.95"
    "gist       32 500 2000 2 10 1000 2346 469 2000 testing 0.99"

    # glove
    "glove      32 500 2000 2 10 10000 533 106 2000 testing 0.8"
    "glove      32 500 2000 2 10 10000 1127 225 2000 testing 0.85"
    "glove      32 500 2000 2 10 10000 1127 225 2000 testing 0.9"
    "glove      32 500 2000 2 10 10000 2845 569 2000 testing 0.95"
    "glove      32 500 2000 2 10 10000 2845 569 2000 testing 0.99"

    # laion
    "laion      32 500 2500 2 10 10000 199 39 2500 testing 0.8"
    "laion      32 500 2500 2 10 10000 278 55 2500 testing 0.85"
    "laion      32 500 2500 2 10 10000 278 55 2500 testing 0.9"
    "laion      32 500 2500 2 10 10000 522 104 2500 testing 0.95"
    # "laion      32 500 2500 2 10 10000 522 104 2500 testing 0.99"

    # t2i
    "t2i        32 500 500  2 10 10000 85 17 500 testing 0.8"
    "t2i        32 500 500  2 10 10000 112 22 500 testing 0.85"
    "t2i        32 500 500  2 10 10000 112 22 500 testing 0.9"
    "t2i        32 500 500  2 10 10000 199 39 500 testing 0.95"
  
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