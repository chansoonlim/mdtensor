/**
 * @file
 * @brief Concept utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "core.hpp"

namespace mdtensor::core {

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

} // namespace mdtensor::core
