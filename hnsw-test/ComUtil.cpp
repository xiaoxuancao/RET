#include "ComUtil.h"

#include <unordered_set>
#include <cstdio>
#include <sys/time.h>
#include <algorithm>


RecallStats recall_at_k_stats(
        const faiss::idx_t* gt,
        const faiss::idx_t* retrieved,
        const float* distances,
        int k,
        int nq,
        const faiss::idx_t* gt_all,
        int k_all,
        bool verbose) {
    double total = 0.0;
    std::vector<double> recalls;
    recalls.reserve(nq);
    for (int i = 0; i < nq; ++i) {
        std::unordered_set<faiss::idx_t> gt_set(gt + i * k, gt + (i + 1) * k);
        int matches = 0;

        for (int j = 0; j < k; ++j) {
            if (gt_set.count(retrieved[i * k + j])) {
                matches++;
            }
        }

        double recall = (double)matches / (double)k;
        recalls.push_back(recall);

        if (verbose) {
            printf("Query[%d]: Recall@%d: %.4f. NN: %f NN_id: %ld\n",
                   i,
                   k,
                   recall,
                   distances[i * k],
                   retrieved[i * k]);

            if (gt_all[i * k_all] != retrieved[i * k]) {
                for (int j = 0; j < k_all; j++) {
                    if (gt_all[i * k_all + j] == retrieved[i * k]) {
                        printf("    NN_id found at %d-th position.\n", j + 1);
                        break;
                    }
                }
            }
        }

        total += recall;
    }

    double avg = total / (double)(nq);
    std::sort(recalls.begin(), recalls.end());

    auto percentile = [&](double p) -> double {
        if (recalls.empty()) return 0.0;
        double rank = p * (recalls.size() - 1);
        size_t lower = static_cast<size_t>(rank);
        size_t upper = std::min(lower + 1, recalls.size() - 1);
        double frac = rank - lower;
        return recalls[lower] * (1.0 - frac) + recalls[upper] * frac;
    };

    double p1 = percentile(0.01);
    double p5 = percentile(0.05);


    return {avg, p1, p5};
}

double recall_at_k_avg(
        const faiss::idx_t* gt,
        const faiss::idx_t* retrieved,
        const float* distances,
        int k,
        int nq,
        const faiss::idx_t* gt_all,
        int k_all,
        bool verbose) {
    double total = 0.0;
    for (int i = 0; i < nq; ++i) {
        std::unordered_set<faiss::idx_t> gt_set(gt + i * k, gt + (i + 1) * k);
        int matches = 0;

        for (int j = 0; j < k; ++j) {
            if (gt_set.count(retrieved[i * k + j])) {
                matches++;
            }
        }

        double recall = (double)matches / (double)k;

        if (verbose) {
            printf("Query[%d]: Recall@%d: %.4f. NN: %f NN_id: %ld\n",
                   i,
                   k,
                   recall,
                   distances[i * k],
                   retrieved[i * k]);

            if (gt_all[i * k_all] != retrieved[i * k]) {
                for (int j = 0; j < k_all; j++) {
                    if (gt_all[i * k_all + j] == retrieved[i * k]) {
                        printf("    NN_id found at %d-th position.\n", j + 1);
                        break;
                    }
                }
            }
        }

        total += recall;
    }

    return total / (double)(nq);
}

std::vector<double> recall_at_k_per_query(
        const faiss::idx_t* gt,
        const faiss::idx_t* retrieved,
        int k,
        int nq) {

    std::vector<double> recalls(nq, 0.0);

    for (int i = 0; i < nq; ++i) {
        std::unordered_set<faiss::idx_t> gt_set(
            gt + i * k,
            gt + (i + 1) * k
        );

        int matches = 0;
        for (int j = 0; j < k; ++j) {
            if (gt_set.count(retrieved[i * k + j])) {
                matches++;
            }
        }

        recalls[i] = static_cast<double>(matches) / static_cast<double>(k);
    }

    return recalls;
}

double recall_at_k_variance(
        const faiss::idx_t* gt,
        const faiss::idx_t* retrieved,
        int k,
        int nq) {

    
    std::vector<double> recalls =
        recall_at_k_per_query(gt, retrieved, k, nq);

    double mean = 0.0;
    for (double r : recalls) {
        mean += r;
    }
    mean /= static_cast<double>(nq);

  
    double var = 0.0;
    for (double r : recalls) {
        double diff = r - mean;
        var += diff * diff;
    }

    var /= static_cast<double>(nq);

    return var;
}

double percentile(
    std::vector<double>& values,
    double p) {   // p in [0, 100]

    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    double pos = (p / 100.0) * (values.size() - 1);
    size_t idx = static_cast<size_t>(pos);

    return values[idx];
}


RecallDeviationStats compute_under_recall_stats(
    const std::vector<double>& recalls,
    float target_recall) {

    RecallDeviationStats stats{};
    std::vector<double> deviations;

    int nq = recalls.size();
    int under_cnt = 0;
    double sum_dev = 0.0;

    for (double r : recalls) {
        if (r < target_recall) {
            double dev = target_recall - r;
            deviations.push_back(dev);
            sum_dev += dev;
            under_cnt++;
        }
    }

    stats.rate = static_cast<double>(under_cnt) / nq;

    if (!deviations.empty()) {
        stats.avg = sum_dev / deviations.size();
        stats.p1  = percentile(deviations, 1.0);
        stats.p99 = percentile(deviations, 99.0);
    } else {
        stats.avg = stats.p1 = stats.p99 = 0.0;
    }

    return stats;
}

RecallDeviationStats compute_over_recall_stats(
    const std::vector<double>& recalls,
    float target_recall) {

    RecallDeviationStats stats{};
    std::vector<double> deviations;

    int nq = recalls.size();
    int over_cnt = 0;
    double sum_dev = 0.0;

    for (double r : recalls) {
        if (r > target_recall) {
            double dev = r - target_recall;
            deviations.push_back(dev);
            sum_dev += dev;
            over_cnt++;
        }
    }

    stats.rate = static_cast<double>(over_cnt) / nq;

    if (!deviations.empty()) {
        stats.avg = sum_dev / deviations.size();
        stats.p1  = percentile(deviations, 1.0);
        stats.p99 = percentile(deviations, 99.0);
    } else {
        stats.avg = stats.p1 = stats.p99 = 0.0;
    }

    return stats;
}


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
        bool shuffle) {
    float* vecsQ_sample = new float[sample_size * dQ];

    faiss::idx_t* gtQ_sample = new faiss::idx_t[sample_size * k];
    float* gt_dist_sample = new float[sample_size * k];

    faiss::idx_t* indicesQ_sample = new faiss::idx_t[sample_size];

    faiss::idx_t* all_indices = new faiss::idx_t[nQ];
    for (faiss::idx_t i = 0; i < nQ; i++) {
        all_indices[i] = i;
    }

    // if (shuffle) {
    //     std::shuffle(
    //             all_indices,
    //             all_indices + nQ,
    //             std::default_random_engine(SEED));
    // }

    for (faiss::idx_t i = 0; i < sample_size; i++) {
        int idx = all_indices[i];

        for (faiss::idx_t j = 0; j < dQ; j++) {
            vecsQ_sample[i * dQ + j] = vecsQ[idx * dQ + j];
        }

        for (faiss::idx_t j = 0; j < k; j++) {
            gtQ_sample[i * k + j] = gt[idx * k + j];
            gt_dist_sample[i * k + j] = gt_dist[idx * k + j];
        }

        indicesQ_sample[i] = idx;
    }
    
    *vecsQ_out = vecsQ_sample;
    *indicesQ_out = indicesQ_sample;
    *gtQ_out = gtQ_sample;
    *gt_dist_out = gt_dist_sample;

    return;
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}


