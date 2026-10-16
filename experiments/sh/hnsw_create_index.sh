# cmake -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -B build -S .
# make -C build -j faiss
# make -C build -j hnsw_test

#!/bin/bash

echo ""
echo ""
echo "============================="
echo ""
echo ""


dataset_params=(
    #"deep       32 500 1000 100"
    # "glove       32 500 1000 100"
    "laion       32 500 2500 100"
    # "t2i       32 500 2500 100"

)

INDEX_BASE_DIR=/data/
DATASET_BASE_DIR=/data/

mode=no-early-stop
query_type=testing
sample=1000
for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS k<<< "$dataset_param"

    INDEX_DIRECTORY=${INDEX_BASE_DIR}/${ds}/fevcs
    DATASET_DIRECTORY=${DATASET_BASE_DIR}

    # ./build/hnsw-test/hnsw_test \
    /home/zrx15003697770/DARTH/build/hnsw-test/hnsw_test \
        --dataset ${ds} \
        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
        --query-num ${sample} --k ${k} \
        --mode ${mode} \
        --index-filepath ${INDEX_DIRECTORY}/${ds}.M${M}.efC${efC}.index \
        --query-type ${query_type} \
        --dataset-dir-prefix ${DATASET_DIRECTORY} \
        --save-index
done
