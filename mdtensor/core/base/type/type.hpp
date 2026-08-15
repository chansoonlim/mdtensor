/**
 * @file
 * @brief Type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

// TODO: include std headers at each file instead of this header
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "concept.hpp"
#include "kokkos.hpp"
#include "promote_type.hpp"

namespace mdtensor::core {

enum class Backend {
    AUTO,   // Automatically select backend based on input types and sizes
    NATIVE, // Native mdtensor implementation
    SIMD,   // SIMD parallelization

#ifdef MDTENSOR_USE_EIGEN
    EIGEN, // Eigen backend
#endif

#ifdef MDTENSOR_USE_OPENMP
    OPENMP, // CPU multi-processing with OpenMP
#endif
};

enum class CopyMode {
    TRUE,  // CopyMode the input tensor to a new tensor
    FALSE, // Do not copy the input tensor; return a view of the input tensor
    AUTO,  // Automatically determine whether to copy or not based on input
};

} // namespace mdtensor::core
