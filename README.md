
<!-- <h2 align="center">⚠️ Repository under construction. Source code is being cleaned. ⚠️ </h2> -->


<h2 align="center">RET: Robust Early Termination in Approximate Nearest Neighbor Search</h2>

## Overview

RET is a workload-adaptive early termination framework for Approximate Nearest Neighbor (ANN) search. Unlike traditional stopping strategies that rely on workload-specific heuristics or fixed thresholds, RET determines termination points from the intrinsic evolution of the search process itself. The framework is designed to improve efficiency without sacrificing retrieval quality, especially in large-scale similarity search workloads.

RET consists of two key components:

- State-aware Trigger: monitors top-k insertion patterns during the search process and identifies phase boundaries in the retrieval trajectory. It then adapts the frequency of evaluation based on the current search state.
- Recall-Aware Predictor: estimates whether the current search state is likely to satisfy the target quality requirements by leveraging distribution-agnostic topological features.

The repository is organized as follows:

- `experiments/` — scripts for reproducing the reported experiments and evaluation settings.
- `faiss/` — modified FAISS implementation with RET integrated.
- `hnsw-tests/` — drivers that launch HNSW-based retrieval and early-termination evaluations.
- `model/` — trained predictor models used for recall estimation and termination decisions.
- `notebooks_scripts/` — utilities and notebook-driven scripts for preprocessing and model training.
- `requirement/` — required Python dependencies and environment configuration files.

## Reproducing the Experiments

To reproduce the experiments, follow the installation steps above and prepare the datasets listed in the Datasets section. Once the environment is configured, run the scripts under `experiments/` to perform the required search workloads and report the metrics.

### Installation
The installation prerequisites can be found in the FAISS installation instructions, which can be found [here](./docs/INSTALL.md).
To run RET, it is required that LightGBM is installed and is put in /HOME/lightgbm-install.
To compile FAISS with RET, use:
```bash
cmake -B build -S . # For the paper, we used: -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -C build -j faiss # Compiles FAISS with RET
make -C build -j hnsw_test # For the driver code of HNSW-based experiments
```

<!-- cmake -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build -S . -->

To use the python scripts and notebooks, install the required Python packages:
```bash
pip install -r requirements.txt
```


### Datasets
Our evaluation utilized the following datasets: 
* [GIST](http://corpus-texmex.irisa.fr/)
* [DEEP and Text2Image](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search)
* [Glove](https://nlp.stanford.edu/projects/glove/)
* [LAION](https://arxiv.org/abs/2111.02114)

The datasets require a minor preprocessing step to be used with RET. For details, refer to the scripts of [utils](./notebooks_scripts/utils).

### Step 1: Build indexing
```bash
cd experiments
cd sh
nohup stdbuf -oL -eL bash hnsw_create_index.sh > ../log/index.log 2>&1 &
```

### Step 2: Generate training data
```bash
nohup stdbuf -oL -eL bash darth_training_data_generation.sh > ../log/darth/darth_offline.log 2>&1 &
nohup stdbuf -oL -eL bash ret_training_data_generation.sh   > ../log/ret/ret_offline.log   2>&1 &
```

### Step 3: Train predictor models
```bash
python -u ret_training.py   > ../train_log/ret_model.log   2>&1 &
python -u laet_training.py  > ../train_log/laet_model.log  2>&1 &
python -u darth_training.py > ../train_log/darth_model.log 2>&1 &
```

### Step 4: Extract parameters (DARTH / LAET)
```bash
python -u extract_data_stats.py > ../train_log/darth_stat.log 2>&1 &
nohup stdbuf -oL -eL bash hnsw_laet_multiplier.sh > ../log/laet/laet_multiplier.log 2>&1 &
```

### Step 5: Online testing
```bash
nohup stdbuf -oL -eL bash optimal_recall_test.sh   > ../log/ret/optimal_online.log 2>&1 &
nohup stdbuf -oL -eL bash hnsw_ret_test.sh          > ../log/ret/ret_online.log     2>&1 &
nohup stdbuf -oL -eL bash hnsw_darth_test_k100.sh   > ../log/darth/darth_online.log 2>&1 &
nohup stdbuf -oL -eL bash hnsw_laet_test_k100.sh    > ../log/laet/laet_online.log  2>&1 &
```

## About
Our experiments on the HNSW index were conducted based on [DARTH]([https://github.com/MChatzakis/DARTH.git]) code.
This repository contains the implementation and integration of RET in the FAISS library, developed by Facebook Research. 
All original FAISS code and components remain under their respective licenses and rules as selected by the developers. 
Please refer to the FAISS license for details regarding using the original library. 
We do not claim any ownership or rights over the original FAISS library: all rights and acknowledgments are retained by the original authors.


