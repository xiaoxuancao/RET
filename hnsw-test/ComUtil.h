#pragma once

#include <faiss/Index.h>
#include <vector>

struct RecallStats {
    double avg;
    double p1;
    double p5;
};

RecallStats recall_at_k_stats(
        const faiss::idx_t* gt,
        const faiss::idx_t* retrieved,
        const float* distances,
        int k,
        int nq,
        const faiss::idx_t* gt_all,
        int k_all,
        bool verbose = false);


double recall_at_k_avg(
        const faiss::idx_t* gt,
        const faiss::idx_t* retrieved,
        const float* distances,
        int k,
        int nq,
        const faiss::idx_t* gt_all,
        int k_all,
        bool verbose = false);

std::vector<double> recall_at_k_per_query(
        const faiss::idx_t* gt,
        const faiss::idx_t* retrieved,
        int k,
        int nq);

double recall_at_k_variance(
        const faiss::idx_t* gt,
        const faiss::idx_t* retrieved,
        int k,
        int nq);

struct RecallDeviationStats {
    double rate;     // under / over recall rate
    double avg;      // average deviation
    double p1;       // P1 deviation
    double p99;      // P99 deviation
};

RecallDeviationStats compute_under_recall_stats(
    const std::vector<double>& recalls,
    float target_recall);

RecallDeviationStats compute_over_recall_stats(
    const std::vector<double>& recalls,
    float target_recall);

void get_queries(
        float* vecsQ,
        faiss::idx_t nQ,
        int dQ,
        faiss::idx_t* gt,
        float *gt_dist,
        faiss::idx_t k,
        faiss::idx_t sample_size,
        float** vecsQ_out,
        faiss::idx_t** indicesQ_out,
        faiss::idx_t** gtQ_out,
        float** gt_dist_out,
        bool shuffle = false);

double elapsed();

