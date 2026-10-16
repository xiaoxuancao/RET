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
    # "deep       32 500 500  2 100 10000 227  45   500  testing 0.5"
    # "deep       32 500 500  2 100 10000 335  67   500  testing 0.6"
    # "deep       32 500 500  2 100 10000 740  148  500  testing 0.8"
    # "deep       32 500 500  2 100 10000 934  187  500  testing 0.85"
    # "deep       32 500 500  2 100 10000 1226 245  500  testing 0.9"
    # "deep       32 500 500  2 100 10000 1730 346  500  testing 0.95"
    # "deep       32 500 500  2 100 10000 2550 510  500  testing 0.99"
    # "deep       32 500 500  2 100 10000 2922 584  500  testing 1.0"

    # gist
    # "gist       32 500 2000 2 100 10000 1152 230  500  testing 0.7"
    # "gist       32 500 2000 2 100 10000 1375 275  500  testing 0.75"
    "gist       32 500 2000 2 100 1000 1662 332  2000  testing 0.8"
    "gist       32 500 2000 2 100 1000 2065 413  2000  testing 0.85"
    "gist       32 500 2000 2 100 1000 2692 538  2000  testing 0.9"
    "gist       32 500 2000 2 100 1000 3905 781  2000  testing 0.95"
    "gist       32 500 2000 2 100 1000 6600 1320 2000  testing 0.99"
    "gist       32 500 2000 2 100 1000 7863 1573 2000  testing 1.0"

    # glove
    # "glove      32 500 2000 2 100 10000 603  121  2000 testing 0.5"
    # "glove      32 500 2000 2 100 10000 1027 205  2000 testing 0.6"
    # "glove      32 500 2000 2 100 10000 2445 489  2000 testing 0.8"
    # "glove      32 500 2000 2 100 10000 2712 542  2000 testing 0.85"
    # "glove      32 500 2000 2 100 10000 2909 582  2000 testing 0.9"
    # "glove      32 500 2000 2 100 10000 3107 621  2000 testing 0.95"
    # "glove      32 500 2000 2 100 10000 3386 677  2000 testing 0.99"
    # "glove      32 500 2000 2 100 10000 3404 681  2000 testing 1.0"

    # laion
    # "laion      32 500 2500 2 100 10000 221  44   2500 testing 0.5"
    # "laion      32 500 2500 2 100 10000 304  61   2500 testing 0.6"
    # "laion      32 500 2500 2 100 10000 646  129  2500 testing 0.8"
    # "laion      32 500 2500 2 100 10000 802  160  2500 testing 0.85"
    # "laion      32 500 2500 2 100 10000 1053 211  2500 testing 0.9"
    # "laion      32 500 2500 2 100 10000 1518 304  2500 testing 0.95"
    # "laion      32 500 2500 2 100 10000 2424 485  2500 testing 0.99"
    # "laion      32 500 2500 2 100 10000 2993 599  2500 testing 1.0"

    # t2i
    # "t2i        32 500 2000 2 100 10000 133  27   2000 testing 0.5"
    # "t2i        32 500 2000 2 100 10000 184  37   2000 testing 0.6"
    "t2i        32 500 500 2 100 10000 363  73   500 testing 0.8"
    "t2i        32 500 500 2 100 10000 443  89   500 testing 0.85"
    "t2i        32 500 500 2 100 10000 563  113  500 testing 0.9"
    "t2i        32 500 500 2 100 10000 778  156  500 testing 0.95"
    "t2i        32 500 500 2 100 10000 1284 257  500 testing 0.99"
    "t2i        32 500 500 2 100 10000 1786 357  500 testing 1.0"
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