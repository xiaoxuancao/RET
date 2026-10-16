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


mode=optimal-recall-testing
train_queries=1000
experiment_times=1

INDEX_DIRECTORY=/data/zrx/
DATASET_DIRECTORY=/data/zrx/
MODEL_DIRECTORY=/../../model


dataset_params=(  
#   "deep      32 500 500  2 100  10000 0.9"
   #"glove       32 500 2000  2 100  10000 0.9 "
   "laion      32 500 5000 2 100 1000 0.9"
  "t2i      32 500 5000 2 100 1000 0.9"
)

#mkdir -p ${RESULTS_DIRECTORY}/${mode}

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS li k sample target_recall <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    echo ""
    echo "--------------- k=${k} ---------------"
    echo ""
    


    for time in $(seq 1 $experiment_times)
    do

        ./../build/hnsw-test/hnsw_test \
        --dataset ${ds} \
        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
        --query-num ${sample} --k $k \
        --mode ${mode} \
        --index-filepath ${INDEX_DIRECTORY}/${ds}/fevcs/${ds}.M${M}.efC${efC}.index \
        --dataset-dir-prefix ${DATASET_DIRECTORY} \
        --target-recall ${target_recall} \
        --query-type ood-testing \
        --output ../../logs/optimal/${ds}_k${k}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_t${time}.txt 
       
    done

    echo ""
    echo ""
done