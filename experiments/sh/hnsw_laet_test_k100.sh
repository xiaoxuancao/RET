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
# mkdir -p ${RESULTS_DIRECTORY}/${mode}
# query_types=ood-testing

# ds M efC efS li k sample target_recall F multiplier model query_types
dataset_params=(
  #deep
#   "deep      32 500 500  2 100  10000 0.7  450    0.13 500 testing"
#   "deep      32 500 500  2 100  10000 0.75 450    0.16 500 testing"
  # "deep      32 500 500  2 100  10000 0.8  450    0.21 500 testing"
  # "deep      32 500 500  2 100  10000 0.85 450    0.27 500 testing"
  # "deep      32 500 500  2 100  10000 0.9  450    0.35 500 testing"
  # "deep      32 500 500  2 100  10000 0.95 450    0.53 500 testing"
  # "deep      32 500 500  2 100  10000 0.99 450    1.11 500 testing"
#   "deep      32 500 500  2 100  10000 1    450    3 500 testing"
  #gist
  "gist      32 500 2000  2 100  1000 0.8  1260    0.20 2000 testing"
  "gist      32 500 2000  2 100  1000 0.85 1260    0.30 2000 testing"
  "gist      32 500 2000  2 100  1000 0.9  1260    0.45 2000 testing"
  "gist      32 500 2000  2 100  1000 0.95 1260    0.80 2000 testing"
  "gist      32 500 2000  2 100  1000 0.99 1260    1.00 2000 testing"
#   "gist      32 500 2000  2 100  1000 1    1260    3 2000 testing"
  #glove
#   "glove     32 500 2000  2 100  10000 0.7 1362   0.07 2000 testing"
#   "glove     32 500 2000  2 100  10000 0.75 1362   0.12 2000 testing"
#   "glove     32 500 2000  2 100  10000 0.8 1362   0.19 2000 testing"
#   "glove     32 500 2000  2 100  10000 0.85 1362   0.29 2000 testing"
#   "glove     32 500 2000  2 100  10000 0.9 1362   0.45 2000 testing"
#   "glove     32 500 2000  2 100  10000 0.95 1362   0.77 2000 testing" 
#   "glove     32 500 2000  2 100  10000 0.99 1362   2.23 2000 testing"
#   "glove     32 500 2000  2 100  10000 1 1362   3 2000 testing"
#    #laion id
#    "laion      32 500 2500 2 100 10000 0.7 1920 0.01 2500 testing"
#    "laion      32 500 2500 2 100 10000 0.75 1920 0.01 2500 testing"
#    "laion      32 500 2500 2 100 10000 0.8 1920 0.01 2500 testing"
#    "laion      32 500 2500 2 100 10000 0.85 1920 0.01 2500 testing"
#    "laion      32 500 2500 2 100 10000 0.9 1920 0.01 2500 testing"
#    "laion      32 500 2500 2 100 10000 0.95 1920 0.41 2500 testing"
  #  "laion      32 500 2500 2 100 10000 0.99 1920 3 2500 testing"
  #  "laion      32 500 2500 2 100 10000  1 1920 3 2500 testing"
   #    #t2i id
#    "t2i      32 500 500 2 100 10000 0.7 400 0.11 500 testing"
#    "t2i      32 500 500 2 100 10000 0.75 400 0.17 500 testing"
#    "t2i      32 500 500 2 100 10000 0.8 400 0.23 500 testing"
#    "t2i      32 500 500 2 100 10000 0.85 400 0.31 500 testing"
#    "t2i      32 500 500 2 100 10000 0.9 400 0.39 500 testing"
#    "t2i      32 500 500 2 100 10000 0.95 400 0.55 500 testing"
#    "t2i      32 500 500 2 100 10000 0.99 400 0.93 500 testing" 
#    "t2i      32 500 500 2 100 10000  1 400 3 500 testing"
#    #laion ood
#    "laion     32 500 3000 2 100 1000 0.7 1920 0.01 3000 ood-testing"
#    "laion      32 500 3000 2 100 1000 0.75 1920 0.01 3000 ood-testing"
#    "laion      32 500 3000 2 100 1000 0.8 1920 0.15 3000 ood-testing"
#    "laion      32 500 3000 2 100 1000 0.85 1920 0.25 3000 ood-testing"
#    "laion      32 500 3000 2 100 1000 0.9 1920 0.43 3000 ood-testing"
#    "laion      32 500 3000 2 100 1000 0.95 1920 1.41 3000 ood-testing"
#    "laion      32 500 3000 2 100 1000 0.99 1920 3 3000 ood-testing"
#    "laion      32 500 3000 2 100 1000  1 1920 3 3000 ood-testing"
#    #t2i ood
#    "t2i      32 500 2500 2 100 1000 0.7 400 0.13 2500 ood-testing"
#    "t2i      32 500 2500 2 100 1000 0.75 400 0.15 2500 ood-testing"
#    "t2i      32 500 2500 2 100 1000 0.8 400 0.17 2500 ood-testing"
#    "t2i      32 500 2500 2 100 1000 0.85 400 0.25 2500 ood-testing"
#    "t2i      32 500 2500 2 100 1000 0.9 400 0.31 2500 ood-testing"
#    "t2i      32 500 2500 2 100 1000 0.95 400 0.55 2500 ood-testing"
#    "t2i      32 500 2500 2 100 1000 0.99 400 1.5 2500 ood-testing" 
#    "t2i      32 500 2500 2 100 1000  1 400 3 2500 ood-testing"

# "laion      32 500 3000 2 100 1000 0.7  1920 0.01 2500 ood-testing"
# "laion      32 500 3000 2 100 1000 0.75 1920 0.01 2500 ood-testing"
# "laion      32 500 3000 2 100 1000 0.8  1920 0.01 2500 ood-testing"
# "laion      32 500 3000 2 100 1000 0.85 1920 0.01 2500 ood-testing"
# "laion      32 500 3000 2 100 1000 0.9  1920 0.01 2500 ood-testing"
# "laion      32 500 3000 2 100 1000 0.95 1920 1.41 2500 ood-testing"
# "laion      32 500 3000 2 100 1000 0.99 1920 3    2500 ood-testing"
# "laion      32 500 3000 2 100 1000 1    1920 3    2500 ood-testing"


# "t2i        32 500 2500  2 100 1000 0.7  400  0.11 500  ood-testing"
# "t2i        32 500 2500  2 100 1000 0.75 400  0.17 500  ood-testing"
# "t2i        32 500 2500  2 100 1000 0.8  400  0.23 500  ood-testing"
# "t2i        32 500 2500  2 100 1000 0.85 400  0.31 500  ood-testing"
# "t2i        32 500 2500  2 100 1000 0.9  400  0.39 500  ood-testing"
# "t2i        32 500 2500  2 100 1000 0.95 400  0.55 500  ood-testing"
# "t2i        32 500 2500  2 100 1000 0.99 400  0.93 500  ood-testing"
# "t2i        32 500 2500  2 100 1000 1    400  3    500  ood-testing"


)

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS li k sample target_recall F multiplier model query_types <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo "--------------- target_recall=${target_recall}, model=${model} ---------------"
    echo "--------------- efS=${efS}, multiplier=${multiplier}, query_type=${query_types}---------------"

   
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
                --query-type ${query_types} \
                --predictor-model-path ${MODEL_DIRECTORY}/laet/${ds}_M${M}_efC${efC}_efS${model}_s${train_queries}_k${k}.txt \
                --output ../../logs/laet/${ds}_${query_types}_k${k}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_t${time}_m${model}.txt 
            #   --output ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_t${time}.txt \ #disable logging

    done


    echo ""
    echo ""
done