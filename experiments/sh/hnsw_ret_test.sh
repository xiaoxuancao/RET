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

mode=ret-testing
n_estim=100
experiment_times=1

INDEX_DIRECTORY=/data/
DATASET_DIRECTORY=/data/
MODEL_DIRECTORY=../../model

declare -A delta_by_recall=(
    [0.8]="5"
    [0.85]="7"
    [0.9]="10"
    [0.95]="15"
    [0.99]="20"
)



target_recall_list=(0.8 0.85 0.9 0.95 0.99)

#testing training_num 
# training_num_list=(100 500 1000 5000 None)
training_num_list=(None)

#parameter, ds M efC efS li k sample tau model query_type
dataset_params=(
    # Noisy query files contain 1K vectors, so sample must not exceed 1000.
    "deep       32 500 500  1 100 1000  1 500  noisy-testing"
    "gist         32 500 2000 1 100 1000  3 2000 noisy-testing"
    "glove      32 500 2000 1 100 1000  3 2000 noisy-testing"
    "laion      32 500 2500 1 100 1000  4 2500 noisy-testing"
    "t2i        32 500 500  1 100 1000  1 500  noisy-testing"
    # # ood
    # "laion      32 500 3000 1 100 1000  5 3000 ood-testing"
    # "t2i        32 500 2500 1 100 1000  4 2500 ood-testing"
    # #robust
    # "laion      32 500 3000 1 100 1000 5 2500 ood-testing"
    # "t2i        32 500 2500 1 100 1000 4 500  ood-testing"
    # "laion      32 500 2500 1 100 10000  4 3000 testing"
    # "t2i        32 500 500  1 100 10000  1 2500 testing"
)

#mkdir -p ${RESULTS_DIRECTORY}/${mode}

for dataset_base in "${dataset_params[@]}"
do
    read ds M efC efS li k sample tau model query_type <<< "$dataset_base" 
    
        for training_num in "${training_num_list[@]}"
        do
            for target_recall in "${target_recall_list[@]}"
            do
            
                delta="${delta_by_recall[$target_recall]}"
                echo "--------------- ${ds} ---------------"
                echo "--------------- target_recall=${target_recall} ---------------"
                echo "--------------- training_num=${training_num} ---------------"
                echo "--------------- delta=${delta} ---------------"
                echo "--------------- efS=${efS}, model=${model}, query_type=${query_type}---------------"

        
            # mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}
            # mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/
            # mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/detailed/
            # mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy

            #--per-prediction-logging \

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
                    --logging-interval ${li} \
                    --delta ${delta} \
                    --tau ${tau} \
                    --query-type ${query_type} \
                    --gnoise-perc 1 \
                    --predictor-model-path ${MODEL_DIRECTORY}/ret/${ds}_M${M}_efC${efC}_efS${model}_k${k}_nestim${n_estim}_tn${training_num}.txt \
                    --output ../../logs/ret/${ds}_${query_type}_k${k}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_delta${delta}_tau${tau}_t${time}_m${model}_tn${training_num}.txt
                done
                # python result_merger.py \
                #       ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval} \
                #       ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}.txt \
                #       $experiment_times
                echo ""
                echo ""
            done
        done
done