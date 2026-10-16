# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# add_executable(hnsw_test EXCLUDE_FROM_ALL hnsw_test.cpp ComUtil.cpp VectorDataLoader.cpp)
# target_link_libraries(hnsw_test PRIVATE faiss)

# add_executable(ivf_test EXCLUDE_FROM_ALL ivf_test.cpp ComUtil.cpp VectorDataLoader.cpp)
# target_link_libraries(ivf_test PRIVATE faiss)

find_package(OpenMP REQUIRED)

find_package(OpenMP REQUIRED)

find_library(OPENBLAS_LIB openblas HINTS ${CONDA_PREFIX}/lib REQUIRED)
find_library(GOMP_LIB gomp HINTS ${CONDA_PREFIX}/lib REQUIRED)

add_executable(hnsw_test EXCLUDE_FROM_ALL
    hnsw_test.cpp
    ComUtil.cpp
    VectorDataLoader.cpp
)

# add_executable(hnsw_test 
#     hnsw_test.cpp
#     ComUtil.cpp
#     VectorDataLoader.cpp
# )

target_link_libraries(hnsw_test
    PRIVATE
        faiss
        OpenMP::OpenMP_CXX
        ${OPENBLAS_LIB}
        ${GOMP_LIB}
)

set_target_properties(hnsw_test PROPERTIES
    BUILD_RPATH "${CONDA_PREFIX}/lib"
)
