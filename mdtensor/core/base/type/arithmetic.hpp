/**
 * @file
 * @brief Arithmetic type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <type_traits>

namespace mdtensor::core {

template <typename T> constexpr bool is_bool_v = std::is_same_v<T, bool>;

template <typename T>
constexpr bool is_non_bool_integral_v = std::is_integral_v<T> && !is_bool_v<T>;

template <typename T>
constexpr bool is_non_bool_unsigned_integral_v =
    is_non_bool_integral_v<T> && std::is_unsigned_v<T>;

template <typename T>
constexpr bool is_non_bool_signed_integral_v =
    is_non_bool_integral_v<T> && std::is_signed_v<T>;

} // namespace mdtensor::core
