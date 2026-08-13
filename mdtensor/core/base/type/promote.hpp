/**
 * @file
 * @brief Numpy-like type promotion for arithmetic types
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace mdtensor::core {
namespace detail {

template <typename T> constexpr bool is_bool_v = std::is_same_v<T, bool>;

template <typename T>
constexpr bool is_non_bool_integral_v = std::is_integral_v<T> && !is_bool_v<T>;

template <typename T>
constexpr bool is_non_bool_unsigned_integral_v =
    is_non_bool_integral_v<T> && std::is_unsigned_v<T>;

template <typename T>
constexpr bool is_non_bool_signed_integral_v =
    is_non_bool_integral_v<T> && std::is_signed_v<T>;

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

template <typename... Ts> struct promote_type_impl;

template <typename T>
    requires(std::is_arithmetic_v<T>)
struct promote_type_impl<T> {
    using type = T;
};

template <typename B1, typename B2>
    requires(is_bool_v<B1> && is_bool_v<B2>)
struct promote_type_impl<B1, B2> {
    // bool + bool -> bool
    using type = bool;
};

template <typename B, typename I>
    requires(is_bool_v<B> && is_non_bool_integral_v<I>)
struct promote_type_impl<B, I> {
    // any integral + bool -> long double
    using type = I;
};

template <typename I, typename B>
    requires(is_non_bool_integral_v<I> && is_bool_v<B>)
struct promote_type_impl<I, B> : promote_type_impl<B, I> {};

template <typename I1, typename I2>
    requires(is_non_bool_integral_v<I1> && is_non_bool_integral_v<I2> &&
             std::is_signed_v<I1> == std::is_signed_v<I2>)
struct promote_type_impl<I1, I2> {
    using type = std::common_type_t<I1, I2>;
};

template <typename S, typename U>
    requires(is_non_bool_signed_integral_v<S> &&
             is_non_bool_unsigned_integral_v<U> && sizeof(U) <= 4)
struct promote_type_impl<S, U> {
    // when one signed and one unsigned type are provided,
    // use a signed type with size at least twice the size of the unsigned type
    using type =
        std::conditional_t<(sizeof(S) >= sizeof(U) * 2), S,
                           typename signed_by_size<sizeof(U) * 2>::type>;
};

template <typename S, typename U>
    requires(is_non_bool_signed_integral_v<S> &&
             is_non_bool_unsigned_integral_v<U> && sizeof(U) > 4)
struct promote_type_impl<S, U> {
    // promote to double when unsigned is 32-bit and signed is less than 64-bit
    using type = double;
};

template <typename U, typename S>
    requires(is_non_bool_unsigned_integral_v<U> &&
             is_non_bool_signed_integral_v<S>)
struct promote_type_impl<U, S> : promote_type_impl<S, U> {};

template <typename I, typename F>
    requires(std::is_integral_v<I> && std::is_floating_point_v<F>)
struct promote_type_impl<I, F> {
    // bool/int8/uint8/int16/uint16 + float       -> float
    // int32/uint32/int64/uint64    + float       -> double
    // any integral                 + double      -> double
    // any integral                 + long double -> long double
    static constexpr bool promote_float_to_double =
        std::is_same_v<F, float> &&
        (std::numeric_limits<I>::digits > std::numeric_limits<float>::digits);

  public:
    using type = std::conditional_t<promote_float_to_double, double, F>;
};

template <typename F, typename I>
    requires(std::is_floating_point_v<F> && std::is_integral_v<I>)
struct promote_type_impl<F, I> : promote_type_impl<I, F> {};

template <typename F1, typename F2>
    requires(std::is_floating_point_v<F1> && std::is_floating_point_v<F2>)
struct promote_type_impl<F1, F2> {
    using type = std::common_type_t<F1, F2>;
};

template <typename T1, typename T2, typename... Ts>
    requires(sizeof...(Ts) > 0 && std::is_arithmetic_v<T1> &&
             std::is_arithmetic_v<T2> && (std::is_arithmetic_v<Ts> && ...))
struct promote_type_impl<T1, T2, Ts...> {
    using type =
        typename promote_type_impl<typename promote_type_impl<T1, T2>::type,
                                   Ts...>::type;
};

} // namespace detail

// NOTE: promote_type_t follows numpy's type promotion rules
template <typename... Ts>
    requires(sizeof...(Ts) > 0 &&
             (std::is_arithmetic_v<std::remove_cvref_t<Ts>> && ...))
using promote_type_t =
    typename detail::promote_type_impl<std::remove_cvref_t<Ts>...>::type;

} // namespace mdtensor::core
