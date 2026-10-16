#!/usr/bin/env bash
# cd ..
# cmake -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -B build -S .
# make -C build -j faiss
# make -C build -j hnsw_test
# cd experiments

echo ""
echo ""
echo "============================="
echo ""
echo ""

INDEX_DIRECTORY=/data/
DATASET_DIRECTORY=/data/
MODEL_DIRECTORY=../../model

train_queries=10000
experiment_times=1

mode=laet-early-stop-testing


multiplier_list=$(seq 0.01 0.01 0.1)

# ds M efC efS li k sample target_recall F query_type
dataset_params=(
  "deep      32 500 500  2 10  1000 0.9 450  validation"
#   "gist      32 500 2000  2 10 1000 0.9 1260  validation"
  "glove     32 500 2000  2 10  1000 0.9 1362  validation"
  "laion     32 500 2500 2 10 1000 0.9 1920  validation"
  "t2i       32 500 500 2 10 1000 0.9 400   validation"
#   "laion      32 500 3000 2 10 1000 0.9 4208  ood-validation"
#   "t2i      32 500 2500 2 10 1000 0.9 400   ood-validation"

  
)

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS li k sample target_recall F query_type <<< "$dataset_param"
    
    for multiplier in $multiplier_list
    do
        echo ""
        echo "--------------- ${ds} ---------------"
        echo "--------------- target_recall=${target_recall} ---------------"
        echo "--------------- efS=${efS}, multiplier=${multiplier}, query_type=${query_type}---------------"

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
                    --fixed-amount-of-search ${F} --prediction-multiplier ${multiplier} \
                    --query-type ${query_type} \
                    --predictor-model-path ${MODEL_DIRECTORY}/laet/${ds}_M${M}_efC${efC}_efS${efS}_s${train_queries}_k${k}.txt \
                    --output ../../logs/laet/${ds}_${query_type}_k${k}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_mult${multiplier}_t${time}.txt 
        done
    done

    echo ""
    echo ""
done