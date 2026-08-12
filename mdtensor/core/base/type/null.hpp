/**
 * @file
 * @brief Nullopt type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <optional>
#include <type_traits>

namespace mdtensor::core {

namespace detail {

template <typename T> struct is_nullopt_impl : std::false_type {};

template <> struct is_nullopt_impl<std::nullopt_t> : std::true_type {};

} // namespace detail

template <typename T> struct is_nullopt_t : detail::is_nullopt_impl<T> {};

template <typename T> constexpr bool is_nullopt_t_v = is_nullopt_t<T>::value;

} // namespace mdtensor::core
