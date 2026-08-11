/**
 * @file
 * @brief Core utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

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

// TODO: Remove when C++23 std::mdspan supports
#ifndef MDSPAN_SINGLE_HEADER_INCLUDE_GUARD_ // for godbolt test
#include <experimental/mdarray>
#include <experimental/mdspan>
#endif

namespace mdtensor {

#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
#define REAL_GCC __GNUC__ // probably
#endif

// TODO: modify under define
#if defined(_OPENMP) && defined(REAL_GCC)
#define MDTENSOR_USE_OPENMP
#endif

namespace core {
namespace stdex = std::experimental;
} // namespace core

} // namespace mdtensor
