#include "DeclarativeRecall.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace faiss {

// DeclarativeRecallDataManager
DeclarativeRecallDataManager::DeclarativeRecallDataManager() {}

DeclarativeRecallDataManager::DeclarativeRecallDataManager(
    float* distances,
    idx_t* labels,
    idx_t* gt,
    float* gt_dist,
    idx_t nq,
    idx_t d,
    idx_t k,
    const float* queries,
    char* log_filename,
    const float* db,
    idx_t ndb,
    idx_t* gt_for_all_k,
    idx_t k_all)
    : distances(distances),
      labels(labels),
      gt(gt),
      gt_dist(gt_dist),
      nq(nq),
      ndb(ndb),
      k(k),
      d(d),
      queries(queries),
      log_filename(log_filename),
      db(db),
      gt_for_all_k(gt_for_all_k),
      k_all(k_all) {}

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float DeclarativeRecallDataManager::get_ed(const float* x, const float* y, size_t d) {
  size_t i;
  float res = 0;
  FAISS_PRAGMA_IMPRECISE_LOOP
  for (i = 0; i < d; i++) {
    const float tmp = x[i] - y[i];
    res += tmp * tmp;
  }
  return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

double DeclarativeRecallDataManager::elapsed_secs() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

float DeclarativeRecallDataManager::get_avg_dist_of_query(idx_t q_id) {
  float sum = 0;
  for (idx_t i = 0; i < k; i++) {
    sum += distances[q_id * k + i];
  }
  return sum / k;
}

float DeclarativeRecallDataManager::get_nearest_dist_of_query(idx_t q_id) {
  float min_dist = std::numeric_limits<float>::max();
  for (idx_t i = 0; i < k; i++) {
    if (distances[q_id * k + i] < min_dist)
      min_dist = distances[q_id * k + i];
  }
  return min_dist;
}

float DeclarativeRecallDataManager::get_furthest_dist_of_query(idx_t q_id) {
  //float max_dist = std::numeric_limits<float>::min();
  float max_dist = std::numeric_limits<float>::lowest();
  for (idx_t i = 0; i < k; i++) {
    if (distances[q_id * k + i] > max_dist)
      max_dist = distances[q_id * k + i];
  }
  return max_dist;
}

float DeclarativeRecallDataManager::get_dist_of_query_to_medoid(idx_t q_id) {
  std::vector<float> medoid(static_cast<size_t>(d), 0.0f);

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < d; j++) {
      medoid[static_cast<size_t>(j)] += db[labels[q_id * k + i] * d + j];
    }
  }

  for (int i = 0; i < d; i++) {
    medoid[static_cast<size_t>(i)] /= k;
  }

  const float* query = queries + q_id * d;

  float ed_dist = get_ed(query, medoid.data(), d);

  return ed_dist;
}

float DeclarativeRecallDataManager::get_recallk(idx_t query_idx) {
  std::unordered_set<idx_t> gt_set(
      gt + query_idx * k, gt + (query_idx + 1) * k);
  int matches = 0;

  for (int j = 0; j < k; ++j) {
    if (gt_set.count(labels[query_idx * k + j])) {
      matches++;
    }
  }

  float recall = (float)matches / (float)k;
  return recall;
}

float DeclarativeRecallDataManager::get_kth_nearest_dist_of_query(idx_t q_id, int kth) {
  float* query_distances = distances + q_id * k;

  std::vector<float> query_distances_cp(query_distances, query_distances + k);

  std::nth_element(
      query_distances_cp.begin(),
      query_distances_cp.begin() + kth,
      query_distances_cp.end());

  float kth_nearest_dist = query_distances_cp[static_cast<size_t>(kth)];

  return kth_nearest_dist;
}

float DeclarativeRecallDataManager::get_lid_of_query(idx_t q_id) {
  const float eps = 1e-12f;
  float max_dist = 0.0f;

  for (idx_t i = 0; i < k; i++) {
    float dist = distances[q_id * k + i];
    if (dist > max_dist) {
      max_dist = dist;
    }
  }

  max_dist = std::max(max_dist, eps);

  float log_sum = 0.0f;

  for (idx_t i = 0; i < k; i++) {
    float dist = distances[q_id * k + i];
    float ratio = std::max(dist / max_dist, eps);
    log_sum += std::log(ratio);
  }

  float log_mean = log_sum / k;

  return -1.0f / log_mean;
}

// float DeclarativeRecallDataManager::get_lid_of_query_percentile(idx_t q_id) {
//   const float eps = 1e-12f;

//   static const float percentiles[10] = {
//       0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
//       0.6f, 0.7f, 0.8f, 0.9f
//   };

//   // static const float percentiles[10] = {
//   //     0.2f, 0.3f, 0.4f, 0.5f,
//   //     0.6f, 0.7f, 0.8f, 0.9f, 1.0f
//   // };
  
//   std::vector<float> samples;
//   samples.reserve(11); 

//   //get nearest
//   float min_dist = std::numeric_limits<float>::max();
//   for (idx_t i = 0; i < k; i++) {
//     min_dist = std::min(min_dist, distances[q_id * k + i]);
//   }
//   samples.push_back(std::max(min_dist, eps));
  
//   //get percentiles
//   for (float p : percentiles) {
//     samples.push_back(
//         std::max(get_percentile_of_query(q_id, p), eps));
//   }

//   //get furthest
//   float max_dist = std::max(
//       get_percentile_of_query(q_id, 1.0f), eps);
//   samples.push_back(max_dist);

//   float log_sum = 0.0f;
//   for (float d : samples) {
//       log_sum += std::log(d / max_dist);
//     }

//     float log_mean = log_sum / samples.size();

//     if (std::abs(log_mean) < 1e-6f) {
//       return 0.0f;  
//     }

//     return -1.0f / log_mean;
// }

float DeclarativeRecallDataManager::get_lid_of_query_percentile(idx_t q_id) {
  const float eps = 1e-12f;

  static const float percentiles[10] = {
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
      0.6f, 0.7f, 0.8f, 0.9f
  };
  
  std::vector<float> samples;
  samples.reserve(11); 

  
  auto to_cosine_distance = [eps](float dist) {
    if (dist >= 0) {
      return std::max(dist, eps);
    }
    
    // similarities = -distances
    float similarity = -dist;
    //similarity = std::max(-1.0f, std::min(1.0f, similarity));
    // cosine_dist = 1 - similarity
    float cosine_dist = 1.0f - similarity;
    return std::max(cosine_dist, eps);
  };


  float min_dist = std::numeric_limits<float>::max();
  for (idx_t i = 0; i < k; i++) {
    float dist = to_cosine_distance(distances[q_id * k + i]);
    min_dist = std::min(min_dist, dist);
  }
  samples.push_back(min_dist);
  

  for (float p : percentiles) {
    float dist = get_percentile_of_query(q_id, p);
    samples.push_back(to_cosine_distance(dist));
  }


  float max_dist = to_cosine_distance(get_percentile_of_query(q_id, 1.0f));
  samples.push_back(max_dist);


  float log_sum = 0.0f;
  for (float d : samples) {
    float ratio = d / max_dist;
    ratio = std::max(ratio, eps);
    log_sum += std::log(ratio);
  }

  float log_mean = log_sum / samples.size();

  if (std::abs(log_mean) < 1e-6f) {
    return 0.0f; 
  }

  return -1.0f / log_mean;
}

float DeclarativeRecallDataManager::get_lid_of_query_percentile_improve(
    float nn_dist,
    float furthest_dist,
    float p1, float p2, float p3, float p4, float p5,
    float p6, float p7, float p8, float p9) {
    
    constexpr float eps = 1e-12f;
    

    auto to_cosine_distance = [eps](float dist) {
        if (dist >= 0) {
            return std::max(dist, eps);
        }
        float similarity = -dist;
        float cosine_dist = 1.0f - similarity;
        return std::max(cosine_dist, eps);
    };
    
  
    float min_dist = to_cosine_distance(nn_dist);
    float max_dist = to_cosine_distance(furthest_dist);
    
    float dist_p1 = to_cosine_distance(p1);
    float dist_p2 = to_cosine_distance(p2);
    float dist_p3 = to_cosine_distance(p3);
    float dist_p4 = to_cosine_distance(p4);
    float dist_p5 = to_cosine_distance(p5);
    float dist_p6 = to_cosine_distance(p6);
    float dist_p7 = to_cosine_distance(p7);
    float dist_p8 = to_cosine_distance(p8);
    float dist_p9 = to_cosine_distance(p9);
    
   
    float samples[] = {min_dist, dist_p1, dist_p2, dist_p3, dist_p4,
                       dist_p5, dist_p6, dist_p7, dist_p8, dist_p9, max_dist};
    
 
    float log_sum = 0.0f;
    for (int i = 0; i < 11; ++i) {
        float ratio = samples[i] / max_dist;
        if (ratio < eps) ratio = eps;
        log_sum += std::log(ratio);
    }
    
    float log_mean = log_sum / 11.0f;
    
    if (std::abs(log_mean) < 1e-6f) {
        return 0.0f;
    }
    return -1.0f / log_mean;
}


float DeclarativeRecallDataManager::get_variance_of_query(idx_t q_id) {
  float sum = 0;
  float sum_of_squares = 0;

  for (idx_t i = 0; i < k; i++) {
    float dist = distances[q_id * k + i];
    sum += dist;
    sum_of_squares += dist * dist;
  }

  float mean = sum / k;

  return (sum_of_squares / k) - (mean * mean);
}

float DeclarativeRecallDataManager::get_percentile_of_query(idx_t q_id, float percentile) {
  //std::vector<float> query_distances(static_cast<size_t>(k));

  // for (idx_t i = 0; i < k; i++) {
  //   query_distances[static_cast<size_t>(i)] = distances[q_id * k + i];
  // }
   std::vector<float> query_distances(
    distances + q_id * k, 
    distances + (q_id + 1) * k
  );

  idx_t idx = static_cast<idx_t>(percentile * (k - 1));

  std::nth_element(
      query_distances.begin(),
      query_distances.begin() + idx,
      query_distances.end());

  return query_distances[static_cast<size_t>(idx)];
}

float DeclarativeRecallDataManager::get_skewness_of_query(idx_t q_id) {
  float sum = 0;
  float sum_of_cubes = 0;
  float sum_of_squares = 0;

  for (idx_t i = 0; i < k; i++) {
    float dist = distances[q_id * k + i];
    sum += dist;
    sum_of_squares += dist * dist;
    sum_of_cubes += dist * dist * dist;
  }

  float mean = sum / k;
  float variance = (sum_of_squares / k) - (mean * mean);
  float skewness = (sum_of_cubes / k) - (3 * mean * variance) - (mean * mean * mean);

  return skewness;
}

float DeclarativeRecallDataManager::get_energy_of_query(idx_t q_id) {
  float sum = 0;

  for (idx_t i = 0; i < k; i++) {
    float dist = distances[q_id * k + i];
    sum += dist * dist;
  }

  return sum;
}

float DeclarativeRecallDataManager::get_kurtosis_of_query(idx_t q_id) {
  float mean = 0;
  float m2 = 0;  // Second moment (variance)
  float m4 = 0;  // Fourth moment
  idx_t n = k;

  // Calculate the mean
  for (idx_t i = 0; i < n; i++) {
    mean += distances[q_id * k + i];
  }
  mean /= n;

  // Calculate second and fourth moments
  for (idx_t i = 0; i < n; i++) {
    float dist = distances[q_id * k + i];
    float diff = dist - mean;
    m2 += diff * diff;
    m4 += diff * diff * diff * diff;
  }

  m2 /= n;  // Variance
  m4 /= n;  // Fourth moment

  // Calculate kurtosis
  float kurtosis = (m4 / (m2 * m2)) - 3;

  return kurtosis;
}

float DeclarativeRecallDataManager::get_TDR(idx_t q_id) {
  float found_distances_sum = 0;
  float gt_distances_sum = 0;

  for (idx_t i = 0; i < k; i++) {
    found_distances_sum += distances[q_id * k + i];
    gt_distances_sum += gt_dist[q_id * k + i];
  }

  return gt_distances_sum / found_distances_sum;
}

float DeclarativeRecallDataManager::get_RDE(idx_t q_id) {
  std::vector<float> found_distances(static_cast<size_t>(k));
  for (idx_t i = 0; i < k; i++) {
    found_distances[static_cast<size_t>(i)] = distances[q_id * k + i];
  }

  // sort by increasing distance to the query
  std::sort(found_distances.begin(), found_distances.end());

  float RDE = 0;

  for (idx_t i = 0; i < k; i++) {
    RDE += (found_distances[static_cast<size_t>(i)] / gt_dist[q_id * k + i]) - 1;
  }

  RDE /= k;

  return RDE;
}

float DeclarativeRecallDataManager::get_NRS(idx_t q_id) {
  std::vector<std::pair<float, idx_t>> distance_label_pairs;

  // Collect approximate distances and labels
  for (int i = 0; i < k; ++i) {
    distance_label_pairs.emplace_back(
        distances[q_id * k + i], labels[q_id * k + i]);
  }

  // Sort the pairs based on distances
  std::sort(
      distance_label_pairs.begin(),
      distance_label_pairs.end(),
      [](const std::pair<float, idx_t>& a,
         const std::pair<float, idx_t>& b) {
        return a.first < b.first;
      });

  // Compute the rank sum for the ground-truth neighbors
  float rank_sum = 0;
  for (int i = 0; i < k; i++) {
    idx_t current_nn_id = distance_label_pairs[static_cast<size_t>(i)].second;

    // Find the rank of the current nearest neighbor in the ground truth
    idx_t gt_position = -1;
    for (int j = 0; j < k_all; j++) {
      if (gt_for_all_k[q_id * k_all + j] == current_nn_id) {
        gt_position = j + 1;  // rank is 1-based
        break;
      }
    }

    // If a ground-truth neighbor is missing, return -1 (undefined)
    if (gt_position == -1) {
      return -1.0;
    }

    rank_sum += gt_position;
  }

  // Calculate max and min possible rank sums
  float max_rank_sum = k * (k + 1) / 2.0;

  // Normalize the rank sum
  float NRS = 1 - ((rank_sum - k) / (max_rank_sum - k));

  return NRS;
}

// DeclarativeRecallDataCollectorHNSW
DeclarativeRecallDataCollectorHNSW::DeclarativeRecallDataCollectorHNSW() {}

DeclarativeRecallDataCollectorHNSW::DeclarativeRecallDataCollectorHNSW(
    DeclarativeRecallDataManager data_manager,
    int logging_interval,
    int dist_early_stop_threshold,
    float target_recall_threshold)
    : data_manager(data_manager),
      logging_interval(logging_interval),
      dist_early_stop_threshold(dist_early_stop_threshold),
      target_recall_threshold(target_recall_threshold) {
        
   if (dist_early_stop_threshold > 0) {
       do_naive_early_stop = true;
  }
  if (target_recall_threshold != 0.00) {
    do_target_recall_stop = true;
  }
}

// DeclarativeRecallDataCollectorIVF
void DeclarativeRecallDataCollectorIVF::init_log_file() {
  if (!dataManager.log_filename) {
    return;
  }

  log_file = fopen(dataManager.log_filename, "w");
  if (!log_file) {
    // printf("Error opening recall data log file %s\n",
    // dataManager.log_filename);
    perror("Error opening recall data log file");
    exit(1);
  }

  fprintf(log_file, "qid,");
  fprintf(log_file, "step,");
  fprintf(log_file, "dists,");
  fprintf(log_file, "elaps_ms,");
  fprintf(log_file, "inserts,");

  fprintf(log_file, "first_nn_dist,");

  fprintf(log_file, "nn_dist,");
  fprintf(log_file, "avg_dist,");
  fprintf(log_file, "furthest_dist,");

  fprintf(log_file, "percentile_25,");
  fprintf(log_file, "percentile_50,");
  fprintf(log_file, "percentile_75,");
  fprintf(log_file, "percentile_95,");

  fprintf(log_file, "variance,");

  // New
  fprintf(log_file, "std,");
  fprintf(log_file, "range,");
  fprintf(log_file, "skewness,");
  fprintf(log_file, "kurtosis,");
  fprintf(log_file, "energy,");
  // New end

  fprintf(log_file, "nn10_dist,");
  fprintf(log_file, "nn_to_first,");
  fprintf(log_file, "nn10_to_first,");

  // Quality approximation measures
  fprintf(log_file, "RDE,");  // relative distance error
  fprintf(log_file, "TDR,");  // total distance ratio
  fprintf(log_file, "NRS,");  // normalized rank sum

  /*if (include_data_dimensions){
      for (int i = 0; i < dataManager.d; i++) {
          fprintf(log_file, "d%d,", i);
      }
  }*/
  fprintf(log_file, "feats_collect_time_ms,");

  // Target
  fprintf(log_file, "r\n");
}

void DeclarativeRecallDataCollectorIVF::close_log_file() {
  if (log_file) {
    fclose(log_file);
  }
}

void DeclarativeRecallDataCollectorIVF::append_to_log(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    float recall_k) {
  if (!log_file || total_insertions < dataManager.k) {
    return;
  }

  // Standard values
  fprintf(log_file, "%ld,", query_idx);
  fprintf(log_file, "%d,", nstep);
  fprintf(log_file, "%d,", ndis);
  fprintf(log_file, "%f,", elapsed * 1000);
  fprintf(log_file, "%d,", total_insertions);
  fprintf(log_file, "%f,", first_nn_dis);

  double feature_collection_time_start = dataManager.elapsed_secs();

  // version 1 for distances
  float nn_dist = dataManager.get_nearest_dist_of_query(query_idx);
  float avg_dist = dataManager.get_avg_dist_of_query(query_idx);
  float furthest_dist = dataManager.get_furthest_dist_of_query(query_idx);

  fprintf(log_file, "%f,", nn_dist);
  fprintf(log_file, "%f,", avg_dist);
  fprintf(log_file, "%f,", furthest_dist);

  float percentile_25 =
      dataManager.get_percentile_of_query(query_idx, 0.25);
  float percentile_50 =
      dataManager.get_percentile_of_query(query_idx, 0.50);
  float percentile_75 =
      dataManager.get_percentile_of_query(query_idx, 0.75);
  float percentile_95 =
      dataManager.get_percentile_of_query(query_idx, 0.95);

  fprintf(log_file, "%f,", percentile_25);
  fprintf(log_file, "%f,", percentile_50);
  fprintf(log_file, "%f,", percentile_75);
  fprintf(log_file, "%f,", percentile_95);

  float variance = dataManager.get_variance_of_query(query_idx);
  fprintf(log_file, "%f,", variance);

  // New includes start
  float stdv = std::sqrt(variance);
  float range = furthest_dist - nn_dist;
  float skewness = dataManager.get_skewness_of_query(query_idx);
  float kurtosis = dataManager.get_kurtosis_of_query(query_idx);
  float energy = dataManager.get_energy_of_query(query_idx);
  fprintf(log_file, "%f,", stdv);
  fprintf(log_file, "%f,", range);
  fprintf(log_file, "%f,", skewness);
  fprintf(log_file, "%f,", kurtosis);
  fprintf(log_file, "%f,", energy);
  // New includes end

  // CMU features
  float dist_10 = -1;
  float dnn_to_dstart = -1;
  float d10_to_dstart = -1;

  if (dataManager.k >= 10) {
    dist_10 = dataManager.get_kth_nearest_dist_of_query(query_idx, 9);
  }

  if (first_nn_dis > 0) {
    dnn_to_dstart = nn_dist / first_nn_dis;
  }

  if (dist_10 != -1 && first_nn_dis > 0) {
    d10_to_dstart = dist_10 / first_nn_dis;
  }

  fprintf(log_file, "%f,", dist_10);
  fprintf(log_file, "%f,", dnn_to_dstart);
  fprintf(log_file, "%f,", d10_to_dstart);
  //

  // version 2 for distances
  // int all_result_set_feats = 11;
  // float dist_feats[all_result_set_feats];
  // get_rset_feats_of_query(query_idx, dist_feats, first_nn_dis);

  // for (int i = 0; i < all_result_set_feats; i++) {
  //     fprintf(log_file, "%f,", dist_feats[i]);
  // }

  // distance from query to medoid
  // float distance_from_medoid =
  //        dataManager.get_dist_of_query_to_medoid(query_idx);
  // fprintf(log_file, "%f,", distance_from_medoid);
  //

  float RDE = dataManager.get_RDE(query_idx);
  float TDR = dataManager.get_TDR(query_idx);
  float NRS = dataManager.get_NRS(query_idx);
  fprintf(log_file, "%f,", RDE);
  fprintf(log_file, "%f,", TDR);
  fprintf(log_file, "%f,", NRS);

  // if (include_data_dimensions){
  //    for (int i = 0; i < dataManager.d; i++) {
  //       fprintf(log_file, "%f,", dataManager.queries[query_idx * dataManager.d + i]);
  //    }
  //}

  // float query_dim_stats[dataManager.summary_stats_num];
  // dataManager.get_precomputed_query_stats(query_idx, query_dim_stats);
  // for (int i = 0; i < dataManager.summary_stats_num; i++) {
  //     fprintf(log_file, "%f,", query_dim_stats[i]);
  // }

  double feature_collection_time_end = dataManager.elapsed_secs();
  double feature_collection_time =
      (feature_collection_time_end - feature_collection_time_start) *
      1000;
  fprintf(log_file, "%f,", feature_collection_time);

  // Target
  fprintf(log_file, "%f\n", recall_k);
}

void DeclarativeRecallDataCollectorHNSW::init_log_file() {
  if (!data_manager.log_filename) {
    return;
  }

  log_file = fopen(data_manager.log_filename, "w");
  if (!log_file) {
    // printf("Error opening recall data log file %s\n",
    // dataManager.log_filename);
    perror("Error opening recall data log file");
    exit(1);
  }

  fprintf(log_file, "qid,");
  fprintf(log_file, "step,");
  fprintf(log_file, "dists,");
  fprintf(log_file, "elaps_ms,");
  fprintf(log_file, "inserts,");

  fprintf(log_file, "first_nn_dist,");

  fprintf(log_file, "nn_dist,");
  fprintf(log_file, "avg_dist,");
  fprintf(log_file, "furthest_dist,");

  fprintf(log_file, "percentile_25,");
  fprintf(log_file, "percentile_50,");
  fprintf(log_file, "percentile_75,");
  fprintf(log_file, "percentile_95,");

  fprintf(log_file, "variance,");

  // New
  fprintf(log_file, "std,");
  fprintf(log_file, "range,");
  fprintf(log_file, "skewness,");
  fprintf(log_file, "kurtosis,");
  fprintf(log_file, "energy,");
  // New end

  fprintf(log_file, "nn10_dist,");
  fprintf(log_file, "nn_to_first,");
  fprintf(log_file, "nn10_to_first,");

  // Quality approximation measures
  fprintf(log_file, "RDE,");  // relative distance error
  fprintf(log_file, "TDR,");  // total distance ratio
  fprintf(log_file, "NRS,");  // normalized rank sum

  fprintf(log_file, "feats_collect_time_ms,");

  // Target
  fprintf(log_file, "r\n");
}

void DeclarativeRecallDataCollectorHNSW::init_ret_file_split() {
  if (!data_manager.log_filename) {
    return;
  }

  log_file = fopen(data_manager.log_filename, "w");
  if (!log_file) {
    // printf("Error opening recall data log file %s\n",
    // dataManager.log_filename);
    perror("Error opening recall data log file");
    exit(1);
  }

  fprintf(log_file, "qid,");
  fprintf(log_file, "step,");
  fprintf(log_file, "dists,");
  fprintf(log_file, "elaps_ms,");
  fprintf(log_file, "inserts,");

  fprintf(log_file, "steps_diff,");
  fprintf(log_file, "dists_diff,");

  fprintf(log_file, "start_nn_dist,");
  fprintf(log_file, "1st_nn_dist,");
  fprintf(log_file, "kth_nn_dist,");
  fprintf(log_file, "0.2kth_nn_dist,");
  fprintf(log_file, "0.3kth_nn_dist,");
  fprintf(log_file, "0.4kth_nn_dist,");
  fprintf(log_file, "0.5kth_nn_dist,");
  fprintf(log_file, "0.6kth_nn_dist,");
  fprintf(log_file, "0.7kth_nn_dist,");
  fprintf(log_file, "0.8kth_nn_dist,");
  fprintf(log_file, "0.9kth_nn_dist,");
  fprintf(log_file, "candidate_dist,");

  fprintf(log_file, "feats_collect_time_ms,");

  // Target
  fprintf(log_file, "r\n");
}

void DeclarativeRecallDataCollectorHNSW::init_ret_file() {
  if (!data_manager.log_filename) {
    return;
  }

  log_file = fopen(data_manager.log_filename, "w");
  if (!log_file) {
    // printf("Error opening recall data log file %s\n",
    // dataManager.log_filename);
    perror("Error opening recall data log file");
    exit(1);
  }

  fprintf(log_file, "qid,");
  fprintf(log_file, "step,");
  fprintf(log_file, "dists,");
  fprintf(log_file, "elaps_ms,");
  fprintf(log_file, "inserts,");

  fprintf(log_file, "steps_diff,");
  fprintf(log_file, "dists_diff,");

  fprintf(log_file, "candidate_dist_diff,");
  fprintf(log_file, "no_insert_max_length,");
  fprintf(log_file, "no_insert_count,");

  fprintf(log_file, "kth_nn_start,");
  fprintf(log_file, "first_nn_start,");
  fprintf(log_file, "first_nn_kth,");

  fprintf(log_file, "first_nn_9kth,");
  fprintf(log_file, "first_nn_8kth,");
  fprintf(log_file, "first_nn_7kth,");

  fprintf(log_file, "kth_nn_9kth,");
  fprintf(log_file, "kth_nn_8kth,");
  fprintf(log_file, "kth_nn_7kth,");

  fprintf(log_file, "lid,");

  fprintf(log_file, "feats_collect_time_ms,");

  // Target
  fprintf(log_file, "r\n");
}



void DeclarativeRecallDataCollectorHNSW::close_log_file() {
  if (log_file) {
    fclose(log_file);
  }
}

void DeclarativeRecallDataCollectorHNSW::append_to_log(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    float recall_k) {
  if (!log_file || total_insertions < data_manager.k) {
    return;
  }

  // Standard values
  fprintf(log_file, "%ld,", query_idx);
  fprintf(log_file, "%d,", nstep);
  fprintf(log_file, "%d,", ndis);
  fprintf(log_file, "%f,", elapsed * 1000);
  fprintf(log_file, "%d,", total_insertions);
  fprintf(log_file, "%f,", first_nn_dis);

  double feature_collection_time_start = data_manager.elapsed_secs();

  // version 1 for distances
  float nn_dist = data_manager.get_nearest_dist_of_query(query_idx);
  float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
  float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);

  fprintf(log_file, "%f,", nn_dist);
  fprintf(log_file, "%f,", avg_dist);
  fprintf(log_file, "%f,", furthest_dist);

  float percentile_25 =
      data_manager.get_percentile_of_query(query_idx, 0.25);
  float percentile_50 =
      data_manager.get_percentile_of_query(query_idx, 0.50);
  float percentile_75 =
      data_manager.get_percentile_of_query(query_idx, 0.75);
  float percentile_95 =
      data_manager.get_percentile_of_query(query_idx, 0.95);

  fprintf(log_file, "%f,", percentile_25);
  fprintf(log_file, "%f,", percentile_50);
  fprintf(log_file, "%f,", percentile_75);
  fprintf(log_file, "%f,", percentile_95);

  float variance = data_manager.get_variance_of_query(query_idx);
  fprintf(log_file, "%f,", variance);

  /* New includes start */
  float stdv = std::sqrt(variance);
  float range = furthest_dist - nn_dist;
  float skewness = data_manager.get_skewness_of_query(query_idx);
  float kurtosis = data_manager.get_kurtosis_of_query(query_idx);
  float energy = data_manager.get_energy_of_query(query_idx);
  fprintf(log_file, "%f,", stdv);
  fprintf(log_file, "%f,", range);
  fprintf(log_file, "%f,", skewness);
  fprintf(log_file, "%f,", kurtosis);
  fprintf(log_file, "%f,", energy);
  /* New includes end */

  // CMU features
  float dist_10 = -1;
  float dnn_to_dstart = -1;
  float d10_to_dstart = -1;

  if (data_manager.k >= 10) {
    dist_10 = data_manager.get_kth_nearest_dist_of_query(query_idx, 9);
  }

  //if (first_nn_dis > 0) {
  if (first_nn_dis != 0) {
    dnn_to_dstart = nn_dist / first_nn_dis;
  }

  //if (dist_10 != -1 && first_nn_dis > 0) {
  if (dist_10 != -1 && first_nn_dis != 0) {
    d10_to_dstart = dist_10 / first_nn_dis;
  }

  fprintf(log_file, "%f,", dist_10);
  fprintf(log_file, "%f,", dnn_to_dstart);
  fprintf(log_file, "%f,", d10_to_dstart);
  //

  // version 2 for distances
  // int all_result_set_feats = 11;
  // float dist_feats[all_result_set_feats];
  // get_rset_feats_of_query(query_idx, dist_feats, first_nn_dis);

  // for (int i = 0; i < all_result_set_feats; i++) {
  //     fprintf(log_file, "%f,", dist_feats[i]);
  // }

  // distance from query to medoid
  // float distance_from_medoid =
  //        dataManager.get_dist_of_query_to_medoid(query_idx);
  // fprintf(log_file, "%f,", distance_from_medoid);
  //

  float RDE = data_manager.get_RDE(query_idx);
  float TDR = data_manager.get_TDR(query_idx);
  float NRS = data_manager.get_NRS(query_idx);
  fprintf(log_file, "%f,", RDE);
  fprintf(log_file, "%f,", TDR);
  fprintf(log_file, "%f,", NRS);

  // if (include_data_dimensions){
  //    for (int i = 0; i < dataManager.d; i++) {
  //       fprintf(log_file, "%f,", dataManager.queries[query_idx * dataManager.d + i]);
  //    }
  //}

  double feature_collection_time_end = data_manager.elapsed_secs();
  double feature_collection_time =
      (feature_collection_time_end - feature_collection_time_start) *
      1000;
  fprintf(log_file, "%f,", feature_collection_time);

  // Target
  fprintf(log_file, "%f\n", recall_k);
}

std::string DeclarativeRecallDataCollectorHNSW::get_observation_data_str_ret_split(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
      //get features
    int steps_diff,
    int dists_diff,
    float candidate_dist,
    float recall_k) {
  std::string observation_data = "";

  if (!log_file || total_insertions < data_manager.k) {
    return observation_data;
  }

  double feature_collection_time_start = data_manager.elapsed_secs();

  observation_data += std::to_string(query_idx) + ",";
  observation_data += std::to_string(nstep) + ",";
  observation_data += std::to_string(ndis) + ",";
  observation_data += std::to_string(elapsed * 1000) + ",";
  observation_data += std::to_string(total_insertions) + ",";

  //steps_diff and dists_diff
  observation_data += std::to_string(steps_diff) + ",";
  observation_data += std::to_string(dists_diff) + ",";
  observation_data += std::to_string(first_nn_dis) + ",";
  float nn_dist = data_manager.get_nearest_dist_of_query(query_idx);
  float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
  observation_data += std::to_string(nn_dist) + ",";
  observation_data += std::to_string(furthest_dist) + ",";

  float perc_2 = data_manager.get_percentile_of_query(query_idx, 0.2);
  float perc_3 = data_manager.get_percentile_of_query(query_idx, 0.3);
  float perc_4 = data_manager.get_percentile_of_query(query_idx, 0.4);
  float perc_5 = data_manager.get_percentile_of_query(query_idx, 0.5);
  float perc_6 = data_manager.get_percentile_of_query(query_idx, 0.6);
  float perc_7 = data_manager.get_percentile_of_query(query_idx, 0.7);
  float perc_8 = data_manager.get_percentile_of_query(query_idx, 0.8);
  float perc_9 = data_manager.get_percentile_of_query(query_idx, 0.9);
  observation_data += std::to_string(perc_2) + ",";
  observation_data += std::to_string(perc_3) + ",";
  observation_data += std::to_string(perc_4) + ",";
  observation_data += std::to_string(perc_5) + ",";
  observation_data += std::to_string(perc_6) + ",";
  observation_data += std::to_string(perc_7) + ",";
  observation_data += std::to_string(perc_8) + ",";
  observation_data += std::to_string(perc_9) + ",";

  observation_data += std::to_string(candidate_dist) + ",";


  double feature_collection_time_end = data_manager.elapsed_secs();
  double feature_collection_time =
      (feature_collection_time_end - feature_collection_time_start) *
      1000;
  observation_data += std::to_string(feature_collection_time) + ",";

  observation_data += std::to_string(recall_k) + "\n";

  return observation_data;
}

std::string DeclarativeRecallDataCollectorHNSW::get_observation_data_str_ret(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    //get features
    int steps_diff,
    int dists_diff,
    //no insert feature
    float candidate_dist_diff,
    int no_insert_max_length,
    int no_insert_count,
    float recall_k) {
  std::string observation_data = "";

  if (!log_file || total_insertions < data_manager.k) {
    return observation_data;
  }

  double feature_collection_time_start = data_manager.elapsed_secs();

  observation_data += std::to_string(query_idx) + ",";
  observation_data += std::to_string(nstep) + ",";
  observation_data += std::to_string(ndis) + ",";
  observation_data += std::to_string(elapsed * 1000) + ",";
  observation_data += std::to_string(total_insertions) + ",";

  //steps_diff and dists_diff
  observation_data += std::to_string(steps_diff) + ",";
  observation_data += std::to_string(dists_diff) + ",";
  
  observation_data += std::to_string(candidate_dist_diff) + ",";
  observation_data += std::to_string(no_insert_max_length) + ",";
  observation_data += std::to_string(no_insert_count) + ",";

  float nn_dist = data_manager.get_nearest_dist_of_query(query_idx);
  float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
  float percentile_7kth = data_manager.get_percentile_of_query(query_idx, 0.7);
  float percentile_8kth = data_manager.get_percentile_of_query(query_idx, 0.8);
  float percentile_9kth = data_manager.get_percentile_of_query(query_idx, 0.9);
  float percentile_6kth = data_manager.get_percentile_of_query(query_idx, 0.6);
  float percentile_5kth = data_manager.get_percentile_of_query(query_idx, 0.5);
  float percentile_4kth = data_manager.get_percentile_of_query(query_idx, 0.4);
  float percentile_3kth = data_manager.get_percentile_of_query(query_idx, 0.3);
  float percentile_2kth = data_manager.get_percentile_of_query(query_idx, 0.2);
  float percentile_1kth = data_manager.get_percentile_of_query(query_idx, 0.1);

  constexpr float EPS = 1e-8f;
  if (nn_dist >= 0 && nn_dist < EPS) nn_dist = EPS;
  if (furthest_dist >= 0 && furthest_dist < EPS) furthest_dist = EPS;
  if (percentile_7kth >= 0 && percentile_7kth < EPS) percentile_7kth = EPS;
  if (percentile_8kth >= 0 && percentile_8kth < EPS) percentile_8kth = EPS;
  if (percentile_9kth >= 0 && percentile_9kth < EPS) percentile_9kth = EPS;
  if (percentile_6kth >= 0 && percentile_6kth < EPS) percentile_6kth = EPS;
  if (percentile_5kth >= 0 && percentile_5kth < EPS) percentile_5kth = EPS;
  if (percentile_4kth >= 0 && percentile_4kth < EPS) percentile_4kth = EPS;
  if (percentile_3kth >= 0 && percentile_3kth < EPS) percentile_3kth = EPS;
  if (percentile_2kth >= 0 && percentile_2kth < EPS) percentile_2kth = EPS;
  if (percentile_1kth >= 0 && percentile_1kth < EPS) percentile_1kth = EPS;

  //kth_nn_start, first_nn_start, first_nn_kth
  float kth_nn_start = first_nn_dis / furthest_dist;
  float first_nn_start = first_nn_dis / nn_dist;
  float first_nn_kth = furthest_dist - nn_dist;
  
  //first_nn_9kth, first_nn_8kth, first_nn_7kth
  float first_nn_9kth = percentile_9kth - nn_dist;
  float first_nn_8kth = percentile_8kth - nn_dist;
  float first_nn_7kth = percentile_7kth - nn_dist;

  //kth_nn_9kth, kth_nn_8kth, kth_nn_7kth
  float kth_nn_9kth = furthest_dist - percentile_9kth;
  float kth_nn_8kth = furthest_dist - percentile_8kth;
  float kth_nn_7kth = furthest_dist - percentile_7kth;

  //lid
  //float lid = data_manager.get_lid_of_query_percentile(query_idx);
  float lid = data_manager.get_lid_of_query_percentile_improve(
    nn_dist, furthest_dist,
    percentile_1kth, percentile_2kth, percentile_3kth, 
    percentile_4kth, percentile_5kth, percentile_6kth,
    percentile_7kth, percentile_8kth, percentile_9kth
);

  
  observation_data += std::to_string(kth_nn_start) + ",";
  observation_data += std::to_string(first_nn_start) + ",";
  observation_data += std::to_string(first_nn_kth) + ",";
  observation_data += std::to_string(first_nn_9kth) + ",";
  observation_data += std::to_string(first_nn_8kth) + ",";    
  observation_data += std::to_string(first_nn_7kth) + ",";
  observation_data += std::to_string(kth_nn_9kth) + ",";
  observation_data += std::to_string(kth_nn_8kth) + ",";
  observation_data += std::to_string(kth_nn_7kth) + ",";
  observation_data += std::to_string(lid) + ",";

  double feature_collection_time_end = data_manager.elapsed_secs();
  double feature_collection_time =
      (feature_collection_time_end - feature_collection_time_start) *
      1000;
  observation_data += std::to_string(feature_collection_time) + ",";

  observation_data += std::to_string(recall_k) + "\n";

  return observation_data;
}

std::string DeclarativeRecallDataCollectorHNSW::get_observation_data_str(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    float recall_k) {
  std::string observation_data = "";

  if (!log_file || total_insertions < data_manager.k) {
    return observation_data;
  }

  double feature_collection_time_start = data_manager.elapsed_secs();

  observation_data += std::to_string(query_idx) + ",";
  observation_data += std::to_string(nstep) + ",";
  observation_data += std::to_string(ndis) + ",";
  observation_data += std::to_string(elapsed * 1000) + ",";
  observation_data += std::to_string(total_insertions) + ",";
  observation_data += std::to_string(first_nn_dis) + ",";

  float nn_dist = data_manager.get_nearest_dist_of_query(query_idx);
  float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
  float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
  observation_data += std::to_string(nn_dist) + ",";
  observation_data += std::to_string(avg_dist) + ",";
  observation_data += std::to_string(furthest_dist) + ",";

  float perc_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
  float perc_50 = data_manager.get_percentile_of_query(query_idx, 0.50);
  float perc_75 = data_manager.get_percentile_of_query(query_idx, 0.75);
  float perc_95 = data_manager.get_percentile_of_query(query_idx, 0.95);
  observation_data += std::to_string(perc_25) + ",";
  observation_data += std::to_string(perc_50) + ",";
  observation_data += std::to_string(perc_75) + ",";
  observation_data += std::to_string(perc_95) + ",";

  float variance = data_manager.get_variance_of_query(query_idx);
  observation_data += std::to_string(variance) + ",";

  /* New includes start */
  float stdv = std::sqrt(variance);
  float range = furthest_dist - nn_dist;
  float skewness = data_manager.get_skewness_of_query(query_idx);
  float kurtosis = data_manager.get_kurtosis_of_query(query_idx);
  float energy = data_manager.get_energy_of_query(query_idx);
  observation_data += std::to_string(stdv) + ",";
  observation_data += std::to_string(range) + ",";
  observation_data += std::to_string(skewness) + ",";
  observation_data += std::to_string(kurtosis) + ",";
  observation_data += std::to_string(energy) + ",";
  /* New includes end */

  // LAET features
  float dist_10 = -1;
  float dnn_to_dstart = -1;
  float d10_to_dstart = -1;

  if (data_manager.k >= 10) {
    dist_10 = data_manager.get_kth_nearest_dist_of_query(query_idx, 9);
  }

  //if (first_nn_dis > 0) {
  if (first_nn_dis != 0) {
    dnn_to_dstart = nn_dist / first_nn_dis;
  }

  //if (dist_10 != -1 && first_nn_dis > 0) {
  if (dist_10 != -1 && first_nn_dis != 0) {
    d10_to_dstart = dist_10 / first_nn_dis;
  }

  observation_data += std::to_string(dist_10) + ",";
  observation_data += std::to_string(dnn_to_dstart) + ",";
  observation_data += std::to_string(d10_to_dstart) + ",";

  // add the quality approximation measures
  float RDE = data_manager.get_RDE(query_idx);
  float TDR = data_manager.get_TDR(query_idx);
  float NRS = data_manager.get_NRS(query_idx);
  observation_data += std::to_string(RDE) + ",";
  observation_data += std::to_string(TDR) + ",";
  observation_data += std::to_string(NRS) + ",";

  double feature_collection_time_end = data_manager.elapsed_secs();
  double feature_collection_time =
      (feature_collection_time_end - feature_collection_time_start) *
      1000;
  observation_data += std::to_string(feature_collection_time) + ",";

  observation_data += std::to_string(recall_k) + "\n";

  return observation_data;
}

void DeclarativeRecallDataCollectorHNSW::flush_observation_to_log(std::string observation_data) {
  if (!log_file) {
    return;
  }

  fprintf(log_file, "%s", observation_data.c_str());
}

void DeclarativeRecallDataCollectorHNSW::flush_all_observations_to_log(std::string* observations, int n) {
  if (!log_file) {
    return;
  }

  for (int i = 0; i < n; i++) {
    fprintf(log_file, "%s", observations[i].c_str());
  }
}

// DARTHPredictorHNSW
DARTHPredictorHNSW::DARTHPredictorHNSW(
    DeclarativeRecallDataManager data_manager,
    double target_recall,
    int initial_prediction_interval,
    int min_prediction_interval,
    bool per_prediction_logging,
    char* predictor_model_path)
    : data_manager(data_manager),
      target_recall(target_recall),
      initial_prediction_interval(initial_prediction_interval),
      min_prediction_interval(min_prediction_interval),
      per_prediction_logging(per_prediction_logging) {
  int out_iterations;
  int result = LGBM_BoosterCreateFromModelfile(
      predictor_model_path, &out_iterations, &booster);
  if (result != 0) {
    exit(1);
  }

  LGBM_SetMaxThreads(1);
}

void DARTHPredictorHNSW::init_log_file() {
  if (!data_manager.log_filename) {
    return;
  }

  log_file = fopen(data_manager.log_filename, "w");
  if (!log_file) {
    printf("Error opening recall data log file\n");
    exit(1);
  }

  fprintf(log_file, "qid,");
  fprintf(log_file, "step,");
  fprintf(log_file, "dists,");
  fprintf(log_file, "elaps_ms,");
  fprintf(log_file, "inserts,");

  fprintf(log_file, "first_nn_dist,");

  fprintf(log_file, "nn_dist,");
  fprintf(log_file, "avg_dist,");
  fprintf(log_file, "furthest_dist,");

  fprintf(log_file, "variance,");
  fprintf(log_file, "median,");

  fprintf(log_file, "percentile_25,");
  fprintf(log_file, "percentile_75,");

  fprintf(log_file, "r_current_interval,");
  fprintf(log_file, "r_predictor_calls,");
  fprintf(log_file, "r_predictor_time_ms,");

  // Quality approximation measures
  fprintf(log_file, "RDE,");  // relative distance error
  fprintf(log_file, "TDR,");  // total distance ratio
  fprintf(log_file, "NRS,");  // normalized rank sum

  fprintf(log_file, "r_actual,");
  fprintf(log_file, "r_predicted\n");
}

void DARTHPredictorHNSW::close_log_file() {
  if (log_file) {
    fclose(log_file);
  }
}

float DARTHPredictorHNSW::predict_recall(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    int query_predictor_calls,
    int* prediction_interval,
    double* predictor_time) {
  if (total_insertions < data_manager.k) {
    return 0.0;
  }

  // query_predictor_calls++;

  double predictor_start_time = data_manager.elapsed_secs();

  float nn_dist = data_manager.get_nearest_dist_of_query(query_idx);
  float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
  float avg_dist = data_manager.get_avg_dist_of_query(query_idx);
  float variance = data_manager.get_variance_of_query(query_idx);
  float median = data_manager.get_percentile_of_query(query_idx, 0.50);
  float percentile_25 = data_manager.get_percentile_of_query(query_idx, 0.25);
  float percentile_75 = data_manager.get_percentile_of_query(query_idx, 0.75);

  const int num_feats = 11;
  const double data[11] = {
      (double)nstep,
      (double)ndis,
      (double)total_insertions,
      (double)first_nn_dis,
      nn_dist,
      avg_dist,
      furthest_dist,
      variance,
      median,
      percentile_25,
      percentile_75};

  double out_result[1];
  long int out_len;

  int res = LGBM_BoosterPredictForMatSingleRow(
      booster,
      data,
      C_API_DTYPE_FLOAT64,
      num_feats,
      1,
      C_API_PREDICT_NORMAL,
      0,
      -1,
      "",
      &out_len,
      out_result);

  if (res != 0) {
    printf("Error predicting recall\n");
    exit(1);
  }

  float predicted_recall = (float)out_result[0];
  predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));  // Make sure recall is in [0, 1]

  if (data_manager.k == 1) {  // Special case for k=1 where recall is 0 or 1
    predicted_recall == predicted_recall >= 0.5 ? 1.0f : 0.0f;
  }

  float recall_diff = target_recall - predicted_recall;
  *prediction_interval = min_prediction_interval + (initial_prediction_interval - min_prediction_interval) * recall_diff;

  double predictor_end_time = data_manager.elapsed_secs();
  *predictor_time += predictor_end_time - predictor_start_time;

  // This logging is only for debugging purposes and may be used only when no multithreading is used
  if (log_file && per_prediction_logging) {
    double actual_recall_k = data_manager.get_recallk(query_idx);

    fprintf(log_file, "%ld,", query_idx);
    fprintf(log_file, "%d,", nstep);
    fprintf(log_file, "%d,", ndis);
    fprintf(log_file, "%f,", elapsed * 1000);
    fprintf(log_file, "%d,", total_insertions);

    fprintf(log_file, "%f,", first_nn_dis);
    fprintf(log_file, "%f,", nn_dist);
    fprintf(log_file, "%f,", avg_dist);
    fprintf(log_file, "%f,", furthest_dist);

    fprintf(log_file, "%f,", variance);
    fprintf(log_file, "%f,", median);

    fprintf(log_file, "%f,", percentile_25);
    fprintf(log_file, "%f,", percentile_75);

    fprintf(log_file, "%d,", *prediction_interval);
    fprintf(log_file, "%d,", query_predictor_calls);
    fprintf(log_file, "%f,", *predictor_time * 1000);

    fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
    fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
    fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

    fprintf(log_file, "%f,", actual_recall_k);
    fprintf(log_file, "%f\n", predicted_recall);
  }

  return predicted_recall;
}

void DARTHPredictorHNSW::log_final_recall_result(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    double last_predicted_recall,
    int prediction_interval,
    int query_predictor_calls,
    double predictor_time) {
  if (!log_file || per_prediction_logging) {
    return;
  }

  fprintf(log_file, "%ld,", query_idx);
  fprintf(log_file, "%d,", nstep);
  fprintf(log_file, "%d,", ndis);
  fprintf(log_file, "%f,", elapsed * 1000);
  fprintf(log_file, "%d,", total_insertions);

  fprintf(log_file, "%f,", first_nn_dis);

  fprintf(log_file, "%f,", data_manager.get_nearest_dist_of_query(query_idx));
  fprintf(log_file, "%f,", data_manager.get_avg_dist_of_query(query_idx));
  fprintf(log_file, "%f,", data_manager.get_furthest_dist_of_query(query_idx));
  fprintf(log_file, "%f,", data_manager.get_variance_of_query(query_idx));
  fprintf(log_file, "%f,", data_manager.get_percentile_of_query(query_idx, 0.50));
  fprintf(log_file, "%f,", data_manager.get_percentile_of_query(query_idx, 0.25));
  fprintf(log_file, "%f,", data_manager.get_percentile_of_query(query_idx, 0.75));

  fprintf(log_file, "%d,", prediction_interval);
  fprintf(log_file, "%d,", query_predictor_calls);
  fprintf(log_file, "%f,", predictor_time * 1000);

  fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
  fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
  fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

  fprintf(log_file, "%f,", data_manager.get_recallk(query_idx));
  fprintf(log_file, "%f\n", last_predicted_recall);
}

// RETPredictorHNSW
RETPredictorHNSW::RETPredictorHNSW(
    DeclarativeRecallDataManager data_manager,
    double target_recall,
    int delta,
    int tau,
    bool per_prediction_logging,
    char* predictor_model_path)
    : data_manager(data_manager),
      target_recall(target_recall),
      delta(delta),
      tau(tau),
      // initial_prediction_interval(initial_prediction_interval),
      // min_prediction_interval(min_prediction_interval),
      per_prediction_logging(per_prediction_logging) {
  int out_iterations;
  int result = LGBM_BoosterCreateFromModelfile(
      predictor_model_path, &out_iterations, &booster);
  if (result != 0) {
    exit(1);
  }

  LGBM_SetMaxThreads(1);
}

void RETPredictorHNSW::init_log_file() {
  //printf("init RET log");
  if (!data_manager.log_filename) {
    return;
  }

  log_file = fopen(data_manager.log_filename, "w");
  if (!log_file) {
    printf("Error opening recall data log file\n");
    exit(1);
  }

  fprintf(log_file, "qid,");
  fprintf(log_file, "step,");
  fprintf(log_file, "dists,");
  fprintf(log_file, "elaps_ms,");
  fprintf(log_file, "inserts,");

  fprintf(log_file, "steps_diff,");
  fprintf(log_file, "dists_diff,");
  
  fprintf(log_file, "candidate_dist_diff,");
  fprintf(log_file, "no_insert_max_length,");
  fprintf(log_file, "no_insert_count,");


  // fprintf(log_file, "first_nn_dist,");
  // fprintf(log_file, "nn_dist,");
  // fprintf(log_file, "furthest_dist,");
  
  fprintf(log_file, "kth_nn_start,");
  fprintf(log_file, "first_nn_start,");

  fprintf(log_file, "first_nn_kth,");

  fprintf(log_file, "first_nn_9kth,");
  fprintf(log_file, "first_nn_8kth,");
  fprintf(log_file, "first_nn_7kth,");

  fprintf(log_file, "kth_nn_9kth,");
  fprintf(log_file, "kth_nn_8kth,");
  fprintf(log_file, "kth_nn_7kth,");

  fprintf(log_file, "lid,");
  
  //fprintf(log_file, "prediction_interval,");
  fprintf(log_file, "r_predictor_calls,");
  fprintf(log_file, "r_predictor_time_ms,");

  // Quality approximation measures
  fprintf(log_file, "RDE,");  // relative distance error
  fprintf(log_file, "TDR,");  // total distance ratio
  fprintf(log_file, "NRS,");  // normalized rank sum

  fprintf(log_file, "r_actual,");
  fprintf(log_file, "r_predicted\n");
}

void RETPredictorHNSW::close_log_file() {
  if (log_file) {
    fclose(log_file);
  }
}

float RETPredictorHNSW::predict_recall(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    int query_predictor_calls,
    //int prediction_interval,
    int steps_diff,
    int dists_diff,
    //no insert feature
    float candidate_dist_diff,
    int no_insert_max_length,
    int no_insert_count,
    double predictor_time){
    // int* prediction_interval,
    // double* predictor_time) {
  if (total_insertions < data_manager.k) {
    return 0.0;
  }
  //printf("RET predict recalls");
  // query_predictor_calls++;

  double predictor_start_time = data_manager.elapsed_secs();

  //double step_diff = 0;
  // double dists_diff = 0;


  float nn_dist = data_manager.get_nearest_dist_of_query(query_idx);
  float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);
  // float nn_dist = data_manager.get_percentile_of_query(query_idx, 0.0);
  // float furthest_dist = data_manager.get_percentile_of_query(query_idx, 1.0);

  float percentile_7kth = data_manager.get_percentile_of_query(query_idx, 0.7);
  float percentile_8kth = data_manager.get_percentile_of_query(query_idx, 0.8);
  float percentile_9kth = data_manager.get_percentile_of_query(query_idx, 0.9);
  float percentile_6kth = data_manager.get_percentile_of_query(query_idx, 0.6);
  float percentile_5kth = data_manager.get_percentile_of_query(query_idx, 0.5);
  float percentile_4kth = data_manager.get_percentile_of_query(query_idx, 0.4);
  float percentile_3kth = data_manager.get_percentile_of_query(query_idx, 0.3);
  float percentile_2kth = data_manager.get_percentile_of_query(query_idx, 0.2);
  float percentile_1kth = data_manager.get_percentile_of_query(query_idx, 0.1);

  constexpr float EPS = 1e-8f;
  if (nn_dist >= 0 && nn_dist < EPS) nn_dist = EPS;
  if (furthest_dist >= 0 && furthest_dist < EPS) furthest_dist = EPS;
  if (percentile_7kth >= 0 && percentile_7kth < EPS) percentile_7kth = EPS;
  if (percentile_8kth >= 0 && percentile_8kth < EPS) percentile_8kth = EPS;
  if (percentile_9kth >= 0 && percentile_9kth < EPS) percentile_9kth = EPS;
  if (percentile_6kth >= 0 && percentile_6kth < EPS) percentile_6kth = EPS;
  if (percentile_5kth >= 0 && percentile_5kth < EPS) percentile_5kth = EPS;
  if (percentile_4kth >= 0 && percentile_4kth < EPS) percentile_4kth = EPS;
  if (percentile_3kth >= 0 && percentile_3kth < EPS) percentile_3kth = EPS;
  if (percentile_2kth >= 0 && percentile_2kth < EPS) percentile_2kth = EPS;
  if (percentile_1kth >= 0 && percentile_1kth < EPS) percentile_1kth = EPS;

  //kth_nn_start
  float kth_nn_start = first_nn_dis / furthest_dist;
  //1st_nn_start
  float first_nn_start = first_nn_dis / nn_dist;
  float first_nn_kth = furthest_dist - nn_dist;

  float first_nn_9kth = percentile_9kth - nn_dist;
  float first_nn_8kth = percentile_8kth - nn_dist;
  float first_nn_7kth = percentile_7kth - nn_dist;

  float kth_nn_9kth = furthest_dist - percentile_9kth;
  float kth_nn_8kth = furthest_dist - percentile_8kth;
  float kth_nn_7kth = furthest_dist - percentile_7kth;

  float lid = data_manager.get_lid_of_query_percentile_improve(
    nn_dist, furthest_dist,
    percentile_1kth, percentile_2kth, percentile_3kth, 
    percentile_4kth, percentile_5kth, percentile_6kth,
    percentile_7kth, percentile_8kth, percentile_9kth
);


  const int num_feats = 18;
  const double data[18] = {
      (double)nstep,
      (double)ndis,
      (double)total_insertions,
      
      (double)steps_diff,
      (double)dists_diff,

      candidate_dist_diff,
      (double)no_insert_max_length,
      (double)no_insert_count,

      kth_nn_start,
      first_nn_start,
      first_nn_kth,
      first_nn_9kth,
      first_nn_8kth,
      first_nn_7kth,
      kth_nn_9kth,
      kth_nn_8kth,
      kth_nn_7kth,
      lid,

      };

  double out_result[1];
  long int out_len;

  int res = LGBM_BoosterPredictForMatSingleRow(
      booster,
      data,
      C_API_DTYPE_FLOAT64,
      num_feats,
      1,
      C_API_PREDICT_NORMAL,
      0,
      -1,
      "",
      &out_len,
      out_result);

  if (res != 0) {
    printf("Error predicting recall\n");
    exit(1);
  }

  float predicted_recall = (float)out_result[0];
  predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));  // Make sure recall is in [0, 1]

  if (data_manager.k == 1) {  // Special case for k=1 where recall is 0 or 1
    predicted_recall == predicted_recall >= 0.5 ? 1.0f : 0.0f;
  }

  float recall_diff = target_recall - predicted_recall;
  //ret
  //*prediction_interval = min_prediction_interval + (initial_prediction_interval - min_prediction_interval) * recall_diff;
  //*prediction_interval = 5;
  //prediction_interval = 5;

  double predictor_end_time = data_manager.elapsed_secs();
  //*predictor_time += predictor_end_time - predictor_start_time;
  predictor_time += predictor_end_time - predictor_start_time;

  // This logging is only for debugging purposes and may be used only when no multithreading is used
  if (log_file && per_prediction_logging) {
    double actual_recall_k = data_manager.get_recallk(query_idx);

    fprintf(log_file, "%ld,", query_idx);
    fprintf(log_file, "%d,", nstep);
    fprintf(log_file, "%d,", ndis);
    fprintf(log_file, "%f,", elapsed * 1000);
    fprintf(log_file, "%d,", total_insertions);

    fprintf(log_file, "%d,", steps_diff);
    fprintf(log_file, "%d,", dists_diff);
    fprintf(log_file, "%f,", candidate_dist_diff);
    fprintf(log_file, "%d,", no_insert_max_length);
    fprintf(log_file, "%d,", no_insert_count);

    // fprintf(log_file, "%f,", first_nn_dis);
    // fprintf(log_file, "%f,", nn_dist);  
    // fprintf(log_file, "%f,", furthest_dist);

    fprintf(log_file, "%f,", kth_nn_start);
    fprintf(log_file, "%f,", first_nn_start);

    fprintf(log_file, "%f,", first_nn_kth);
    fprintf(log_file, "%f,", first_nn_9kth);
    fprintf(log_file, "%f,", first_nn_8kth);
    fprintf(log_file, "%f,", first_nn_7kth);

    fprintf(log_file, "%f,", kth_nn_9kth);
    fprintf(log_file, "%f,", kth_nn_8kth);
    fprintf(log_file, "%f,", kth_nn_7kth);
    fprintf(log_file, "%f,",  lid);

    // fprintf(log_file, "%d,", *prediction_interval);
    //fprintf(log_file, "%d,", prediction_interval);
    fprintf(log_file, "%d,", query_predictor_calls);
    // fprintf(log_file, "%f,", *predictor_time * 1000);
    fprintf(log_file, "%f,", predictor_time * 1000);


    fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
    fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
    fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

    fprintf(log_file, "%f,", actual_recall_k);
    fprintf(log_file, "%f\n", predicted_recall);
  }

  return predicted_recall;
}

void RETPredictorHNSW::log_final_recall_result(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    double last_predicted_recall,
    //int prediction_interval,
    int steps_diff,
    int dists_diff,
    //no insert feature
    float candidate_dist_diff,
    int no_insert_max_length,
    int no_insert_count,
    int query_predictor_calls,
    double predictor_time) {
  if (!log_file || per_prediction_logging) {
    return;
  }
  //printf("RET final log");

  fprintf(log_file, "%ld,", query_idx);
  fprintf(log_file, "%d,", nstep);
  fprintf(log_file, "%d,", ndis);
  fprintf(log_file, "%f,", elapsed * 1000);
  fprintf(log_file, "%d,", total_insertions);

  // double step_diff = 0;
  // double dists_diff = 0;

  fprintf(log_file, "%d,", steps_diff);
  fprintf(log_file, "%d,", dists_diff);
  fprintf(log_file, "%f,", candidate_dist_diff);
  fprintf(log_file, "%d,", no_insert_max_length);
  fprintf(log_file, "%d,", no_insert_count);


  float nn_dist = data_manager.get_nearest_dist_of_query(query_idx);
  float furthest_dist = data_manager.get_furthest_dist_of_query(query_idx);

  float percentile_7kth = data_manager.get_percentile_of_query(query_idx, 0.7);
  float percentile_8kth = data_manager.get_percentile_of_query(query_idx, 0.8);
  float percentile_9kth = data_manager.get_percentile_of_query(query_idx, 0.9);
  float percentile_6kth = data_manager.get_percentile_of_query(query_idx, 0.6);
  float percentile_5kth = data_manager.get_percentile_of_query(query_idx, 0.5);
  float percentile_4kth = data_manager.get_percentile_of_query(query_idx, 0.4);
  float percentile_3kth = data_manager.get_percentile_of_query(query_idx, 0.3);
  float percentile_2kth = data_manager.get_percentile_of_query(query_idx, 0.2);
  float percentile_1kth = data_manager.get_percentile_of_query(query_idx, 0.1);

  constexpr float EPS = 1e-8f;
  if (nn_dist >= 0 && nn_dist < EPS) nn_dist = EPS;
  if (furthest_dist >= 0 && furthest_dist < EPS) furthest_dist = EPS;
  if (percentile_7kth >= 0 && percentile_7kth < EPS) percentile_7kth = EPS;
  if (percentile_8kth >= 0 && percentile_8kth < EPS) percentile_8kth = EPS;
  if (percentile_9kth >= 0 && percentile_9kth < EPS) percentile_9kth = EPS;
  if (percentile_6kth >= 0 && percentile_6kth < EPS) percentile_6kth = EPS;
  if (percentile_5kth >= 0 && percentile_5kth < EPS) percentile_5kth = EPS;
  if (percentile_4kth >= 0 && percentile_4kth < EPS) percentile_4kth = EPS;
  if (percentile_3kth >= 0 && percentile_3kth < EPS) percentile_3kth = EPS;
  if (percentile_2kth >= 0 && percentile_2kth < EPS) percentile_2kth = EPS;
  if (percentile_1kth >= 0 && percentile_1kth < EPS) percentile_1kth = EPS;

  float kth_nn_start = first_nn_dis / furthest_dist;
  float first_nn_start = first_nn_dis / nn_dist;
  float first_nn_kth = furthest_dist - nn_dist;
  float first_nn_9kth = percentile_9kth - nn_dist;
  float first_nn_8kth = percentile_8kth - nn_dist;
  float first_nn_7kth = percentile_7kth - nn_dist;
  float kth_nn_9kth = furthest_dist - percentile_9kth;
  float kth_nn_8kth = furthest_dist - percentile_8kth;
  float kth_nn_7kth = furthest_dist - percentile_7kth;
  float lid = data_manager.get_lid_of_query_percentile_improve(
    nn_dist, furthest_dist,
    percentile_1kth, percentile_2kth, percentile_3kth, 
    percentile_4kth, percentile_5kth, percentile_6kth,
    percentile_7kth, percentile_8kth, percentile_9kth
  );

  // fprintf(log_file, "%f,", first_nn_dis);
  // fprintf(log_file, "%f,", nn_dist);  
  // fprintf(log_file, "%f,", furthest_dist);

  fprintf(log_file, "%f,", kth_nn_start);
  fprintf(log_file, "%f,", first_nn_start);

  fprintf(log_file, "%f,", first_nn_kth);
  fprintf(log_file, "%f,", first_nn_9kth);
  fprintf(log_file, "%f,", first_nn_8kth);
  fprintf(log_file, "%f,", first_nn_7kth);
  fprintf(log_file, "%f,", kth_nn_9kth);
  fprintf(log_file, "%f,", kth_nn_8kth);
  fprintf(log_file, "%f,", kth_nn_7kth);
  fprintf(log_file, "%f,",  lid);

  //fprintf(log_file, "%d,", prediction_interval);
  fprintf(log_file, "%d,", query_predictor_calls);
  fprintf(log_file, "%f,", predictor_time * 1000);

  fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
  fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
  fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));

  fprintf(log_file, "%f,", data_manager.get_recallk(query_idx));
  fprintf(log_file, "%f\n", last_predicted_recall);
}

// LAETPredictorHNSW
LAETPredictorHNSW::LAETPredictorHNSW(
    DeclarativeRecallDataManager data_manager,
    int fixed_amount_of_distance_calcs,
    float prediction_multiplier,
    char* predictor_model_path)
    : data_manager(data_manager),
      fixed_amount_of_distance_calcs(fixed_amount_of_distance_calcs),
      prediction_multiplier(prediction_multiplier) {
  int out_iterations;
  int result = LGBM_BoosterCreateFromModelfile(predictor_model_path, &out_iterations, &booster);
  if (result != 0) {
    exit(1);
  }

  LGBM_SetMaxThreads(1);
}

void LAETPredictorHNSW::init_log_file() {
  if (!data_manager.log_filename) {
    return;
  }

  log_file = fopen(data_manager.log_filename, "w");
  if (!log_file) {
    // printf("Error opening recall data log file\n");
    perror("Error opening recall data log file");
    exit(1);
  }

  fprintf(log_file, "qid,");
  fprintf(log_file, "dists,");
  fprintf(log_file, "elaps_ms,");

  fprintf(log_file, "predictor_time,");
  fprintf(log_file, "predicted_distance_calcs,");

  // Quality approximation measures
  fprintf(log_file, "RDE,");  // relative distance error
  fprintf(log_file, "TDR,");  // total distance ratio
  fprintf(log_file, "NRS,");  // normalized rank sum

  fprintf(log_file, "r\n");
}

void LAETPredictorHNSW::close_log_file() {
  if (log_file) {
    fclose(log_file);
  }
}

int LAETPredictorHNSW::predict_distance_calcs(
    idx_t query_idx,
    int nstep,
    int ndis,
    double elapsed,
    float first_nn_dis,
    double* predictor_time) {
  double predictor_start_time = data_manager.elapsed_secs();

  float d_start = first_nn_dis;
  float d_1st = data_manager.get_nearest_dist_of_query(query_idx);
  float d_10th = -1;
  if (data_manager.k >= 10) {
    d_10th = data_manager.get_kth_nearest_dist_of_query(query_idx, 9);
  }

  float d_1st_to_start = -1;
  if (d_start != 0) {
    d_1st_to_start = d_1st / d_start;
  }

  float d_10th_to_start = -1;
  if (d_10th != -1 && d_start != 0) {
    d_10th_to_start = d_10th / d_start;
  }

  int num_nonquery_feats = 5;
  int query_feats = data_manager.d;
  int total_feats = num_nonquery_feats + query_feats;
  std::vector<double> feats(static_cast<size_t>(total_feats));

  feats[0] = d_start;
  feats[1] = d_1st;
  feats[2] = d_10th;
  feats[3] = d_1st_to_start;
  feats[4] = d_10th_to_start;

  for (int i = 0; i < data_manager.d; i++) {
    feats[static_cast<size_t>(num_nonquery_feats + i)] =
        data_manager.queries[query_idx * data_manager.d + i];
  }

  int num_rows = 1;
  int num_feats = total_feats;

  double out_result[1];
  long int out_len;
  int res = LGBM_BoosterPredictForMatSingleRow(
      booster,
      feats.data(),
      C_API_DTYPE_FLOAT64,
      num_feats,
      1,
      C_API_PREDICT_NORMAL,
      0,
      -1,
      "",
      &out_len,
      out_result);

  if (res != 0) {
    printf("Error predicting distance calcs\n");
    exit(1);
  }

  int predicted_distance_calcs = (int)out_result[0];

  predicted_distance_calcs = static_cast<int>(std::round(prediction_multiplier * predicted_distance_calcs));

  double predictor_end_time = data_manager.elapsed_secs();
  *predictor_time += predictor_end_time - predictor_start_time;

  if (per_prediction_logging && log_file) {
    fprintf(log_file, "%ld,", query_idx);
    fprintf(log_file, "%d,", ndis);
    fprintf(log_file, "%f,", elapsed * 1000);
    fprintf(log_file, "%f,", *predictor_time * 1000);
    fprintf(log_file, "%d,", predicted_distance_calcs);
    fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
    fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
    fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));
    fprintf(log_file, "%f\n", data_manager.get_recallk(query_idx));
  }

  return predicted_distance_calcs;
}

void LAETPredictorHNSW::log_final_result(
    idx_t query_idx,
    int nstep,
    int ndis,
    double elapsed,
    float first_nn_dis,
    int predicted_distance_calcs,
    double predictor_time) {
  if (!log_file || per_prediction_logging) {
    return;
  }

  fprintf(log_file, "%ld,", query_idx);
  fprintf(log_file, "%d,", ndis);
  fprintf(log_file, "%f,", elapsed * 1000);
  fprintf(log_file, "%f,", predictor_time * 1000);
  fprintf(log_file, "%d,", predicted_distance_calcs);
  fprintf(log_file, "%f,", data_manager.get_RDE(query_idx));
  fprintf(log_file, "%f,", data_manager.get_TDR(query_idx));
  fprintf(log_file, "%f,", data_manager.get_NRS(query_idx));
  fprintf(log_file, "%f\n", data_manager.get_recallk(query_idx));
}

// DARTHPredictorIVF
DARTHPredictorIVF::DARTHPredictorIVF(
    DeclarativeRecallDataManager dataManager,
    double target_recall,
    int initial_prediction_interval,
    int min_prediction_interval,
    bool per_prediction_logging,
    char* predictor_model_path,
    int logging_interval)
    : dataManager(dataManager),
      target_recall(target_recall),
      initial_prediction_interval(initial_prediction_interval),
      min_prediction_interval(min_prediction_interval),
      per_prediction_logging(per_prediction_logging),
      logging_interval(logging_interval) {
  int out_iterations;
  int result = LGBM_BoosterCreateFromModelfile(
      predictor_model_path, &out_iterations, &booster);
  if (result != 0) {
    exit(1);
  }

  LGBM_SetMaxThreads(1);
}

void DARTHPredictorIVF::init_log_file() {
  if (!dataManager.log_filename) {
    return;
  }

  log_file = fopen(dataManager.log_filename, "w");
  if (!log_file) {
    printf("Error opening recall data log file\n");
    exit(1);
  }

  fprintf(log_file, "qid,");
  fprintf(log_file, "step,");
  fprintf(log_file, "dists,");
  fprintf(log_file, "elaps_ms,");
  fprintf(log_file, "inserts,");

  fprintf(log_file, "first_nn_dist,");

  fprintf(log_file, "nn_dist,");
  fprintf(log_file, "avg_dist,");
  fprintf(log_file, "furthest_dist,");

  fprintf(log_file, "variance,");
  fprintf(log_file, "median,");

  fprintf(log_file, "percentile_25,");
  fprintf(log_file, "percentile_75,");

  fprintf(log_file, "r_current_interval,");
  fprintf(log_file, "r_predictor_calls,");
  fprintf(log_file, "r_predictor_time_ms,");

  // Quality approximation measures
  fprintf(log_file, "RDE,");  // relative distance error
  fprintf(log_file, "TDR,");  // total distance ratio
  fprintf(log_file, "NRS,");  // normalized rank sum

  fprintf(log_file, "r_actual,");
  fprintf(log_file, "r_predicted\n");
}

void DARTHPredictorIVF::close_log_file() {
  if (log_file) {
    fclose(log_file);
  }
}

float DARTHPredictorIVF::predictRecall(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    int query_predictor_calls,
    int* prediction_interval,
    double* predictor_time) {
  if (total_insertions < dataManager.k) {
    return 0.0f;
  }

  query_predictor_calls++;

  double predictor_start_time = dataManager.elapsed_secs();

  float nn_dist = dataManager.get_nearest_dist_of_query(query_idx);
  float furthest_dist = dataManager.get_furthest_dist_of_query(query_idx);
  float avg_dist = dataManager.get_avg_dist_of_query(query_idx);
  float variance = dataManager.get_variance_of_query(query_idx);
  float median = dataManager.get_percentile_of_query(query_idx, 0.50);
  float percentile_25 = dataManager.get_percentile_of_query(query_idx, 0.25);
  float percentile_75 = dataManager.get_percentile_of_query(query_idx, 0.75);

  const int num_feats = 11;
  const double data[11] = {
      (double)nstep,
      (double)ndis,
      (double)total_insertions,
      (double)first_nn_dis,
      nn_dist,
      avg_dist,
      furthest_dist,
      variance,
      median,
      percentile_25,
      percentile_75};

  double out_result[1];
  long int out_len;

  int res = LGBM_BoosterPredictForMatSingleRow(
      booster,
      data,
      C_API_DTYPE_FLOAT64,
      num_feats,
      1,
      C_API_PREDICT_NORMAL,
      0,
      -1,
      "",
      &out_len,
      out_result);

  if (res != 0) {
    printf("Error predicting recall\n");
    exit(1);
  }

  float predicted_recall = (float)out_result[0];
  predicted_recall = std::min(1.0f, std::max(0.0f, predicted_recall));

  if (dataManager.k == 1) {  // Special case for k=1 where recall is 0 or 1
    predicted_recall == predicted_recall >= 0.5 ? 1.0f : 0.0f;
  }

  float recall_diff = static_cast<float>(target_recall) - predicted_recall;
  *prediction_interval = min_prediction_interval +
                         (initial_prediction_interval - min_prediction_interval) *
                             recall_diff;

  *prediction_interval = ((*prediction_interval + logging_interval - 1) / logging_interval) * logging_interval;

  double predictor_end_time = dataManager.elapsed_secs();
  *predictor_time += predictor_end_time - predictor_start_time;

  if (log_file && per_prediction_logging) {
    double actual_recall_k = dataManager.get_recallk(query_idx);

    fprintf(log_file, "%ld,", query_idx);
    fprintf(log_file, "%d,", nstep);
    fprintf(log_file, "%d,", ndis);
    fprintf(log_file, "%f,", elapsed * 1000);
    fprintf(log_file, "%d,", total_insertions);

    fprintf(log_file, "%f,", first_nn_dis);
    fprintf(log_file, "%f,", nn_dist);
    fprintf(log_file, "%f,", avg_dist);
    fprintf(log_file, "%f,", furthest_dist);

    fprintf(log_file, "%f,", variance);
    fprintf(log_file, "%f,", median);

    fprintf(log_file, "%f,", percentile_25);
    fprintf(log_file, "%f,", percentile_75);

    fprintf(log_file, "%d,", *prediction_interval);
    fprintf(log_file, "%d,", query_predictor_calls);
    fprintf(log_file, "%f,", *predictor_time * 1000);

    fprintf(log_file, "%f,", dataManager.get_RDE(query_idx));
    fprintf(log_file, "%f,", dataManager.get_TDR(query_idx));
    fprintf(log_file, "%f,", dataManager.get_NRS(query_idx));

    fprintf(log_file, "%f,", actual_recall_k);
    fprintf(log_file, "%f\n", predicted_recall);
  }

  return predicted_recall;
}

void DARTHPredictorIVF::log_final_recall_result(
    idx_t query_idx,
    int nstep,
    int ndis,
    int total_insertions,
    double elapsed,
    float first_nn_dis,
    double last_predicted_recall,
    int prediction_interval,
    int query_predictor_calls,
    double predictor_time) {
  if (!log_file || per_prediction_logging) {
    return;
  }

  fprintf(log_file, "%ld,", query_idx);
  fprintf(log_file, "%d,", nstep);
  fprintf(log_file, "%d,", ndis);
  fprintf(log_file, "%f,", elapsed * 1000);
  fprintf(log_file, "%d,", total_insertions);

  fprintf(log_file, "%f,", first_nn_dis);

  fprintf(log_file, "%f,", dataManager.get_nearest_dist_of_query(query_idx));
  fprintf(log_file, "%f,", dataManager.get_avg_dist_of_query(query_idx));
  fprintf(log_file, "%f,", dataManager.get_furthest_dist_of_query(query_idx));
  fprintf(log_file, "%f,", dataManager.get_variance_of_query(query_idx));
  fprintf(log_file, "%f,", dataManager.get_percentile_of_query(query_idx, 0.50));
  fprintf(log_file, "%f,", dataManager.get_percentile_of_query(query_idx, 0.25));
  fprintf(log_file, "%f,", dataManager.get_percentile_of_query(query_idx, 0.75));

  fprintf(log_file, "%d,", prediction_interval);
  fprintf(log_file, "%d,", query_predictor_calls);
  fprintf(log_file, "%f,", predictor_time * 1000);

  fprintf(log_file, "%f,", dataManager.get_RDE(query_idx));
  fprintf(log_file, "%f,", dataManager.get_TDR(query_idx));
  fprintf(log_file, "%f,", dataManager.get_NRS(query_idx));

  fprintf(log_file, "%f,", dataManager.get_recallk(query_idx));
  fprintf(log_file, "%f\n", last_predicted_recall);
}

}  // namespace faiss
