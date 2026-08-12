/**
 * @file
 * @brief Type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "common_integral_type.hpp"

namespace mdtensor::core {
namespace detail {

template <typename... Ts> struct common_arithmetic_type_impl {
    // no type defined
};

template <typename T>
    requires(std::is_arithmetic_v<T>)
struct common_arithmetic_type_impl<T> {
    using type = T;
};

template <typename I1, typename I2>
    requires(requires { typename common_integral_type_t<I1, I2>; })
struct common_arithmetic_type_impl<I1, I2> {
    // when two integral types are provided, use common_integral_type_t
    using type = common_integral_type_t<I1, I2>;
};

template <typename I, typename F>
    requires(std::is_integral_v<I> && std::is_floating_point_v<F>)
struct common_arithmetic_type_impl<I, F> {
    // when one integral and one floating-point type are provided,
    // use the floating-point type
    using type = F;
};

template <typename F, typename I>
    requires(std::is_floating_point_v<F> && std::is_integral_v<I>)
struct common_arithmetic_type_impl<F, I> : common_arithmetic_type_impl<I, F> {};

template <typename F1, typename F2>
    requires(std::is_floating_point_v<F1> && std::is_floating_point_v<F2>)
struct common_arithmetic_type_impl<F1, F2> {
    // when two floating-point types are provided, use the type with larger size
    using type = std::conditional_t<(sizeof(F1) >= sizeof(F2)), F1, F2>;
};

template <typename T1, typename T2, typename... Ts>
    requires(sizeof...(Ts) > 0 && std::is_arithmetic_v<T1> &&
             std::is_arithmetic_v<T2> && (std::is_arithmetic_v<Ts> && ...))
struct common_arithmetic_type_impl<T1, T2, Ts...> {
    using type = typename common_arithmetic_type_impl<
        typename common_arithmetic_type_impl<T1, T2>::type, Ts...>::type;
};

} // namespace detail

template <typename... Ts>
    requires(sizeof...(Ts) > 0 &&
             (std::is_arithmetic_v<std::remove_cvref_t<Ts>> && ...))
using common_arithmetic_type_t = typename detail::common_arithmetic_type_impl<
    std::remove_cvref_t<Ts>...>::type;

} // namespace mdtensor::core
