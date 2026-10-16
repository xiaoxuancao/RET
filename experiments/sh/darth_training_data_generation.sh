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

# ds M efC efS  sample query_type
dataset_params=(
    "deep       32 500 500 10000 training"
    "deep       32 500 500 1000 validation"
    "deep       32 500 500 10000 testing"
    "glove       32 500 2000 10000 training"
    "glove       32 500 2000 1000 validation"
    "glove       32 500 2000 10000 testing"
    "gist      32 500 2000 10000 training"
    "gist      32 500 2000 1000 validation"
    "gist      32 500 2000 1000 testing"
    "laion       32 500 2500 10000 training"
    "t2i       32 500 500 10000 training"
    "laion       32 500 2500 1000 validation"
    "t2i       32 500 500 1000 validation"
    "laion       32 500 2500 10000 testing"
    "t2i       32 500 500 10000 testing"
    # #dataset ood
    "laion       32 500 3000 10000 ood-training"
    "t2i       32 500 2500 10000 ood-training"
    "laion       32 500 3000 1000 ood-validation"
    "t2i       32 500 2500 1000 ood-validation"
    "laion       32 500 3000 1000 ood-testing"
    "t2i       32 500 2500 1000 ood-testing"

)

k_values=(10 50)


INDEX_BASE_DIR=/data/
DATASET_BASE_DIR=/data/

mode=early-stop-training


    for dataset_param in "${dataset_params[@]}"
    do
        read ds M efC efS query_num query_type<<< "$dataset_param"

        # mkdir ${TRAINING_DATA_DIRECTORY}/${mode}/${ds}

        for k in ${k_values[@]}
        do
            # mkdir ${TRAINING_DATA_DIRECTORY}/${mode}/${ds}/k${k}
            echo "--------------- ${ds} ---------------"
            echo "--------------- training_num=${query_num} ---------------"
            echo "--------------- efS=${efS}, k=${k}, query_type=${query_type}---------------"

            for logging_interval in 2
            do
                INDEX_DIRECTORY=${INDEX_BASE_DIR}/${ds}/fevcs
                DATASET_DIRECTORY=${DATASET_BASE_DIR}
                TRAINING_DATA_DIRECTORY=${INDEX_BASE_DIR}/darth
                ../../build/hnsw-test/hnsw_test \
                        --dataset ${ds} \
                        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                        --query-num ${query_num} --k $k \
                        --mode ${mode} \
                        --logging-interval ${logging_interval} \
                        --index-filepath ${INDEX_DIRECTORY}/${ds}.M${M}.efC${efC}.index \
                        --dataset-dir-prefix ${DATASET_DIRECTORY} \
                        --query-type ${query_type} \
                        --output ${TRAINING_DATA_DIRECTORY}/${ds}_${query_type}_k${k}_M${M}_efC${efC}_efS${efS}_qs${query_num}_li${logging_interval}.txt
            done
        done


        echo ""
        echo ""
    done


echo "Training data generation completed."

# exit

# Generate the validation data
# We need the detailed data with the target variables to evaluate the model on the val dataset for 1000 queries
# mode=early-stop-training
# query_type=ood-validation
# # mkdir ./results/validation_logging/
# for query_num in 1000
# do
#     for dataset_param in "${dataset_params[@]}"
#     do
#         read ds M efC efS k<<< "$dataset_param"

#         # mkdir ./results/validation_logging/${ds}

#         echo ""
#         echo "--------------- ${ds} ---------------"
#         echo ""

#         # for k in ${k_values[@]}
#         # do
#             # mkdir ./results/validation_logging/${ds}/k${k}

#             for logging_interval in 1
#             do
#                 INDEX_DIRECTORY=${INDEX_BASE_DIR}/${ds}/fevcs
#                 DATASET_DIRECTORY=${DATASET_BASE_DIR}
#                 TRAINING_DATA_DIRECTORY=${INDEX_BASE_DIR}/darth
#                 ../build/hnsw-test/hnsw_test \
#                         --dataset ${ds} \
#                         --query-type ${query_type} \
#                         --M ${M} --efConstruction ${efC} --efSearch ${efS} \
#                         --query-num ${query_num} --k $k \
#                         --output ${TRAINING_DATA_DIRECTORY}/${ds}_${query_type}_k${k}_M${M}_efC${efC}_efS${efS}_qs${query_num}_li${logging_interval}.txt \
#                         --mode ${mode} \
#                         --logging-interval ${logging_interval} \
#                         --index-filepath ${INDEX_DIRECTORY}/${ds}.M${M}.efC${efC}.index \
#                         --dataset-dir-prefix ${DATASET_DIRECTORY} 
                
#             done
#         # done

#         echo ""
#         echo ""
#     done
# done


# # We do the same for the testing data
# # We need this to evaluate the final test performance of the model
# mode=early-stop-training
# # mkdir ./results/test_logging/
# query_type=ood-testing
# for query_num in 1000
# do
#     for dataset_param in "${dataset_params[@]}"
#     do
#         read ds M efC efS k<<< "$dataset_param"

#         # mkdir ./results/test_logging/${ds}

#         echo ""
#         echo "--------------- ${ds} ---------------"
#         echo ""

#         # for k in ${k_values[@]}
#         # do
#             # mkdir ./results/test_logging/${ds}/k${k}

#             for logging_interval in 1
#             do
#                 INDEX_DIRECTORY=${INDEX_BASE_DIR}/${ds}/fevcs
#                 DATASET_DIRECTORY=${DATASET_BASE_DIR}
#                 TRAINING_DATA_DIRECTORY=${INDEX_BASE_DIR}/darth
#                 ../build/hnsw-test/hnsw_test \
#                         --dataset ${ds} \
#                         --M ${M} --efConstruction ${efC} --efSearch ${efS} \
#                         --query-num ${query_num} --k $k \
#                         --output ${TRAINING_DATA_DIRECTORY}/${ds}_${query_type}_k${k}_M${M}_efC${efC}_efS${efS}_qs${query_num}_li${logging_interval}.txt \
#                         --mode ${mode} \
#                         --logging-interval ${logging_interval} \
#                         --index-filepath ${INDEX_DIRECTORY}/${ds}.M${M}.efC${efC}.index \
#                         --dataset-dir-prefix ${DATASET_DIRECTORY} \
#                         --query-type ${query_type}
#             done
#         # done

#         echo ""
#         echo ""
#     done
# done


# exit 




