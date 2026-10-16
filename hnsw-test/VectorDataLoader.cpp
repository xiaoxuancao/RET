#include "VectorDataLoader.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

const char query_type_str[7][20] = {
    "Training",
    "Validation",
    "Testing",
    "OOD Training",
    "OOD Validation",
    "OOD Testing"
};

float* fvecs_read(
        const char* fname,
        size_t* d_out,
        size_t* n_out,
        size_t limit) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        exit(1);
    }

    int d;
    fread(&d, 1, sizeof(int), f);

    assert((d > 0 && d < 1000000) && "unreasonable dimension");

    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 && "weird file size");
    size_t n = sz / ((d + 1) * 4);

    if (limit > 0 && n > limit) {
        n = limit;
        printf("Limiting dataset %s to %ld vectors\n", fname, n);
    }

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) && "could not read whole file");

    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

VectorDataLoader::VectorDataLoader(std::string dataset_name, query_type_t query_type)
    : query_type(query_type), noise_perc(""), dataset_name(dataset_name) {
}

VectorDataLoader::VectorDataLoader(std::string dataset_name, query_type_t query_type, std::string noise_perc, std::string directory_path)
    : VectorDataLoader(dataset_name, query_type) {
    this->directory_path = directory_path;
    this->noise_perc = noise_perc;
}

void VectorDataLoader::initializeDataMaps(){

    // GIST1M
    baseVectorsMap["gist"] = directory_path + "gist/fevcs/base.1M.fvecs";

    queryTypeToVectorsMap[TRAINING]["gist"] = directory_path + "gist/fevcs/learn.100K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["gist"] = directory_path + "gist/fevcs/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["gist"] = directory_path + "gist/fevcs/query.1K.fvecs";
    

    queryTypeToGroundtruthsMap[TRAINING]["gist"] = directory_path + "gist/gt/l2/learn.groundtruth.100K.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["gist"] = directory_path + "gist/gt/l2/validation.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["gist"] = directory_path + "gist/gt/l2/query.groundtruth.1K.k1000.ivecs";
   

    queryTypeToGroundtruthDistancesMap[TRAINING]["gist"] = directory_path + "gist/gt/l2/learn.groundtruth.100K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["gist"] = directory_path + "gist/gt/l2/validation.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["gist"] = directory_path + "gist/gt/l2/query.groundtruth.1K.k1000.fvecs";
 
    // t2i
    baseVectorsMap["t2i"] = directory_path + "t2i/fevcs/base.10M.fvecs";        
    queryTypeToVectorsMap[TRAINING]["t2i"] = directory_path + "t2i/fevcs/learn.10K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["t2i"] = directory_path + "t2i/fevcs/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["t2i"] = directory_path + "t2i/fevcs/query.10K.fvecs";

    

    queryTypeToVectorsMap[OOD_TRAINING]["t2i"] = directory_path + "t2i/fevcs/learn_ood.10K.fvecs";
    queryTypeToVectorsMap[OOD_VALIDATION]["t2i"] = directory_path + "t2i/fevcs/validation_ood.1K.fvecs";
    queryTypeToVectorsMap[OOD_TESTING]["t2i"] = directory_path + "t2i/fevcs/query_ood.10K.fvecs";
    
    // gt
    queryTypeToGroundtruthsMap[TRAINING]["t2i"] = directory_path + "t2i/gt/ip/learn.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["t2i"] = directory_path + "t2i/gt/ip/validation.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["t2i"] = directory_path + "t2i/gt/ip/query.groundtruth.10K.k1000.ivecs";
    
  
    
    queryTypeToGroundtruthsMap[OOD_TRAINING]["t2i"] = directory_path + "t2i/gt/ip/learn_ood.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[OOD_VALIDATION]["t2i"] = directory_path + "t2i/gt/ip/validation_ood.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[OOD_TESTING]["t2i"] = directory_path + "t2i/gt/ip/query_ood.groundtruth.10K.k1000.ivecs";


    //gt distances
    queryTypeToGroundtruthDistancesMap[TRAINING]["t2i"] = directory_path + "t2i/gt/ip/learn.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["t2i"] = directory_path + "t2i/gt/ip/validation.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["t2i"] = directory_path + "t2i/gt/ip/query.groundtruth.10K.k1000.fvecs";

  
    
    queryTypeToGroundtruthDistancesMap[OOD_TRAINING]["t2i"] = directory_path + "t2i/gt/ip/learn_ood.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[OOD_VALIDATION]["t2i"] = directory_path + "t2i/gt/ip/validation_ood.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[OOD_TESTING]["t2i"] = directory_path + "t2i/gt/ip/query_ood.groundtruth.10K.k1000.fvecs";



    // DEEP10M
    baseVectorsMap["deep"] = directory_path + "deep/fevcs/base.9.99M.fvecs";
    
    queryTypeToVectorsMap[TRAINING]["deep"] = directory_path + "deep/fevcs/learn.10K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["deep"] = directory_path + "deep/fevcs/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["deep"] = directory_path + "deep/fevcs/query.10K.fvecs";
    
    queryTypeToGroundtruthsMap[TRAINING]["deep"] = directory_path + "deep/gt/cosine/learn.groundtruth.10K.k100.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["deep"] = directory_path + "deep/gt/cosine/validation.groundtruth.1K.k100.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["deep"] = directory_path + "deep/gt/cosine/query.groundtruth.10K.k100.ivecs";
    

    queryTypeToGroundtruthDistancesMap[TRAINING]["deep"] = directory_path + "deep/gt/cosine/learn.groundtruth.10K.k100.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["deep"] = directory_path + "deep/gt/cosine/validation.groundtruth.1K.k100.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["deep"] = directory_path + "deep/gt/cosine/query.groundtruth.10K.k100.fvecs";
    

    // GLOVE1M
    baseVectorsMap["glove"] = directory_path + "glove/fevcs/base.1.18M.fvecs";
    queryTypeToVectorsMap[TRAINING]["glove"] = directory_path + "glove/fevcs/learn.10K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["glove"] = directory_path + "glove/fevcs/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["glove"] = directory_path + "glove/fevcs/query.10K.fvecs";
    

    queryTypeToGroundtruthsMap[TRAINING]["glove"] = directory_path + "glove/gt/cosine/learn.groundtruth.10K.k100.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["glove"] = directory_path + "glove/gt/cosine/validation.groundtruth.1K.k100.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["glove"] = directory_path + "glove/gt/cosine/query.groundtruth.10K.k100.ivecs";
   
    queryTypeToGroundtruthDistancesMap[TRAINING]["glove"] = directory_path + "glove/gt/cosine/learn.groundtruth.10K.k100.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["glove"] = directory_path + "glove/gt/cosine/validation.groundtruth.1K.k100.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["glove"] = directory_path + "glove/gt/cosine/query.groundtruth.10K.k100.fvecs";
    

    // laion
    baseVectorsMap["laion"] = directory_path + "laion/fevcs/base.30.65M.fvecs";

    queryTypeToVectorsMap[TRAINING]["laion"] = directory_path + "laion/fevcs/learn.10K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["laion"] = directory_path + "laion/fevcs/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["laion"] = directory_path + "laion/fevcs/query.10K.fvecs";
    

    queryTypeToVectorsMap[OOD_TRAINING]["laion"] = directory_path + "laion/fevcs/learn_t2i.10K.fvecs";
    queryTypeToVectorsMap[OOD_VALIDATION]["laion"] = directory_path + "laion/fevcs/validation_t2i.1K.fvecs";
    queryTypeToVectorsMap[OOD_TESTING]["laion"] = directory_path + "laion/fevcs/query_t2i.10K.fvecs";

    //gt
    queryTypeToGroundtruthsMap[TRAINING]["laion"] = directory_path + "laion/gt/cosine/learn.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["laion"] = directory_path + "laion/gt/cosine/validation.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["laion"] = directory_path + "laion/gt/cosine/query.groundtruth.10K.k1000.ivecs";
   

    queryTypeToGroundtruthsMap[OOD_TRAINING]["laion"] = directory_path + "laion/gt/cosine/learn_t2i.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[OOD_VALIDATION]["laion"] = directory_path + "laion/gt/cosine/validation_t2i.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[OOD_TESTING]["laion"] = directory_path + "laion/gt/cosine/query_t2i.groundtruth.10K.k1000.ivecs";

    //gt distances
    queryTypeToGroundtruthDistancesMap[TRAINING]["laion"] = directory_path + "laion/gt/cosine/learn.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["laion"] = directory_path + "laion/gt/cosine/validation.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["laion"] = directory_path + "laion/gt/cosine/query.groundtruth.10K.k1000.fvecs";


    queryTypeToGroundtruthDistancesMap[OOD_TRAINING]["laion"] = directory_path + "laion/gt/cosine/learn_t2i.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[OOD_VALIDATION]["laion"] = directory_path + "laion/gt/cosine/validation_t2i.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[OOD_TESTING]["laion"] = directory_path + "laion/gt/cosine/query_t2i.groundtruth.10K.k1000.fvecs";




}

float* VectorDataLoader::loadDB(size_t* d_out, size_t* n_out) {
    std::string baseVectorsPath = baseVectorsMap[dataset_name];
    
    printf(">> Loading base vectors from: %s\n", baseVectorsPath.c_str());
    float *db = fvecs_read(baseVectorsPath.c_str(), d_out, n_out);

    return db;
}

float* VectorDataLoader::loadQueries(size_t* d_out, size_t* n_out) {
    std::string queryVectorsPath = queryTypeToVectorsMap[query_type][dataset_name];
    
    printf(">> Loading queries from: %s\n", queryVectorsPath.c_str());
    float *queries = fvecs_read(queryVectorsPath.c_str(), d_out, n_out);

    // skip first 5000 queries:
    //*n_out -= 5000;
    //queries += 5000 * *d_out;

    return queries;
}

int* VectorDataLoader::loadQueriesGroundtruths(size_t* k_out, size_t* n_out) {
    std::string queryGroundtruthsPath = queryTypeToGroundtruthsMap[query_type][dataset_name];
    
    printf(">> Loading queries groundtruths from: %s\n", queryGroundtruthsPath.c_str());
    int *gt = ivecs_read(queryGroundtruthsPath.c_str(), k_out, n_out);

    return gt;
}

float* VectorDataLoader::loadQueriesGroundtruthDistances(size_t* k_out, size_t* n_out) {
    std::string queryGroundtruthDistancesPath = queryTypeToGroundtruthDistancesMap[query_type][dataset_name];
    
    printf(">> Loading queries groundtruth distances from: %s\n", queryGroundtruthDistancesPath.c_str());
    float *gt = fvecs_read(queryGroundtruthDistancesPath.c_str(), k_out, n_out);

    return gt;
}


