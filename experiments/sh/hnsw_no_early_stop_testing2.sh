#!/usr/bin/env bash

# $HOME/cmake-4.1.2-linux-x86_64/bin/cmake -B build -S . \
#  -DCMAKE_BUILD_TYPE=Release \
#  -DFAISS_ENABLE_GPU=OFF \
#  -DFAISS_OPT_LEVEL=avx512 \
#  -DBUILD_SHARED_LIBS=ON  \
#  -DCMAKE_CXX_FLAGS_RELEASE="-O3" \
#  -DCMAKE_INSTALL_PREFIX=$HOME/lightgbm-install \
#  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
#  -DCMAKE_CXX_FLAGS_RELEASE="-O3"
# make -C build -j faiss
# make -C build -j hnsw_test

echo ""
echo ""
echo "============================="
echo ""
echo ""

dataset_params=(
    "deep       32 500 500  100  10000 testing"
    "gist      32 500 2000  100  1000 testing"
    "glove       32 500 2000  100  10000 testing"
    "laion       32 500 2500 100 10000 testing"
    "t2i         32 500 2000  100  10000 testing"
    "laion       32 500 3000 100 1000 ood-testing"
    "t2i       32 500 2500 100 1000 ood-testing"

    )

INDEX_DIRECTORY=/data/
DATASET_DIRECTORY=/data/

mode=no-early-stop-testing-2
# mode=no-early-stop

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS k sample query_type<<< "$dataset_param"

    ./../../build/hnsw-test/hnsw_test \
        --dataset ${ds} \
        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
        --query-num ${sample} --k ${k} \
        --mode ${mode} \
        --index-filepath ${INDEX_DIRECTORY}/${ds}/fevcs/${ds}.M${M}.efC${efC}.index \
        --query-type ${query_type} \
        --dataset-dir-prefix ${DATASET_DIRECTORY} \
        --output ../../logs/rem/${ds}_${query_type}_k${k}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_t${time}.txt 
done