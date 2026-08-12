/**
 * @file
 * @brief Common integral type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "arithmetic.hpp"

namespace mdtensor::core {
namespace detail {

template <std::size_t size> struct signed_by_size;

template <> struct signed_by_size<1> {
    using type = std::int8_t;
};

template <> struct signed_by_size<2> {
    using type = std::int16_t;
};

template <> struct signed_by_size<4> {
    using type = std::int32_t;
};

template <> struct signed_by_size<8> {
    using type = std::int64_t;
};

template <typename... Ts> struct common_integral_type_impl {
    // no type defined
};

template <typename T>
    requires(std::is_integral_v<T>)
struct common_integral_type_impl<T> {
    // when one type is provided, use the type itself
    using type = T;
};

template <typename B1, typename B2>
    requires(is_bool_v<B1> && is_bool_v<B2>)
struct common_integral_type_impl<B1, B2> {
    // when two bool types are provided, use bool
    using type = bool;
};

template <typename B, typename I>
    requires(is_bool_v<B> && is_non_bool_integral_v<I>)
struct common_integral_type_impl<B, I> {
    // when one bool and one non-bool integral type are provided,
    // use the non-bool integral type
    using type = I;
};

template <typename I, typename B>
    requires(is_non_bool_integral_v<I> && is_bool_v<B>)
struct common_integral_type_impl<I, B> : common_integral_type_impl<B, I> {};

template <typename I1, typename I2>
    requires(is_non_bool_integral_v<I1> && is_non_bool_integral_v<I2> &&
             std::is_signed_v<I1> == std::is_signed_v<I2>)
struct common_integral_type_impl<I1, I2> {
    // when two types with same signedness are provided,
    // use the type with larger size
    using type = std::conditional_t<(sizeof(I1) >= sizeof(I2)), I1, I2>;
};

template <typename S, typename U>
    requires(is_non_bool_signed_integral_v<S> &&
             is_non_bool_unsigned_integral_v<U> && sizeof(U) <= 4)
struct common_integral_type_impl<S, U> {
    // when one signed and one unsigned type are provided,
    // use a signed type with size at least twice the size of the unsigned type
    using type =
        std::conditional_t<(sizeof(S) >= sizeof(U) * 2), S,
                           typename signed_by_size<sizeof(U) * 2>::type>;
};

template <typename U, typename S>
    requires(is_non_bool_unsigned_integral_v<U> &&
             is_non_bool_signed_integral_v<S> && sizeof(U) <= 4)
struct common_integral_type_impl<U, S> : common_integral_type_impl<S, U> {};

template <typename T1, typename T2, typename... Ts>
    requires(sizeof...(Ts) > 0 && (std::is_integral_v<Ts> && ...))
struct common_integral_type_impl<T1, T2, Ts...> {
    using type = typename common_integral_type_impl<
        typename common_integral_type_impl<T1, T2>::type, Ts...>::type;
};

} // namespace detail

template <typename... Ts>
    requires(sizeof...(Ts) > 0 &&
             (std::is_integral_v<std::remove_cvref_t<Ts>> && ...))
using common_integral_type_t = typename detail::common_integral_type_impl<
    std::remove_cvref_t<Ts>...>::type;

} // namespace mdtensor::core
