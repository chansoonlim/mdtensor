/**
 * @file
 * @brief Type utilities for mdtensor.
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

template <typename T>
concept integral_c = std::integral<std::remove_cvref_t<T>>;

template <typename T>
concept unsigned_integral_c = std::unsigned_integral<std::remove_cvref_t<T>>;

template <typename T>
concept floating_point_c = std::floating_point<std::remove_cvref_t<T>>;

template <typename T>
concept arithmetic_c = std::integral<std::remove_cvref_t<T>> ||
                       std::floating_point<std::remove_cvref_t<T>>;

namespace detail {

template <typename T> struct is_nullopt_impl : std::false_type {};

template <> struct is_nullopt_impl<std::nullopt_t> : std::true_type {};

} // namespace detail

template <typename T> struct is_nullopt_t : detail::is_nullopt_impl<T> {};

template <typename T> constexpr bool is_nullopt_t_v = is_nullopt_t<T>::value;

template <typename T>
concept nullopt_t_c = is_nullopt_t_v<std::remove_cvref_t<T>>;

} // namespace core
} // namespace mdtensor
