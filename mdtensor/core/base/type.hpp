/**
 * @file
 * @brief Type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "core.hpp"

// TODO: change namespace to mdtensor::type
namespace mdtensor::core {

template <typename T> constexpr bool is_bool_v = std::same_as<T, bool>;

template <typename T>
constexpr bool is_non_bool_integral_v = std::integral<T> && !is_bool_v<T>;

template <typename T>
constexpr bool is_non_bool_unsigned_integral_v =
    std::unsigned_integral<T> && !is_bool_v<T>;

template <typename T>
constexpr bool is_non_bool_signed_integral_v =
    std::signed_integral<T> && !is_bool_v<T>;

namespace detail {

template <typename T> struct is_nullopt_impl : std::false_type {};

template <> struct is_nullopt_impl<std::nullopt_t> : std::true_type {};

} // namespace detail

template <typename T> struct is_nullopt_t : detail::is_nullopt_impl<T> {};

template <typename T> constexpr bool is_nullopt_t_v = is_nullopt_t<T>::value;

template <typename T>
concept nullopt_t_c = is_nullopt_t_v<std::remove_cvref_t<T>>;

template <typename T>
concept integral_c = std::integral<std::remove_cvref_t<T>>;

template <typename T>
concept floating_point_c = std::floating_point<std::remove_cvref_t<T>>;

template <typename T>
concept arithmetic_c = std::integral<std::remove_cvref_t<T>> ||
                       std::floating_point<std::remove_cvref_t<T>>;

////////////////////////////////////////////////////////////////
/////////////// common_integral_type_t /////////////////////////
////////////////////////////////////////////////////////////////

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

template <std::integral T> struct common_integral_type_impl<T> {
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

template <std::integral T1, std::integral T2, std::integral... Ts>
    requires(sizeof...(Ts) > 0)
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

////////////////////////////////////////////////////////////////
/////////////// common_arithmetic_type_t ///////////////////////
////////////////////////////////////////////////////////////////

namespace detail {

template <typename... Ts> struct common_arithmetic_type_impl {
    // no type defined
};

template <typename T>
    requires(std::is_arithmetic_v<T>)
struct common_arithmetic_type_impl<T> {
    using type = T;
};

template <std::integral I1, std::integral I2>
    requires(requires { typename common_integral_type_t<I1, I2>; })
struct common_arithmetic_type_impl<I1, I2> {
    // when two integral types are provided, use common_integral_type_t
    using type = common_integral_type_t<I1, I2>;
};

template <std::integral I, std::floating_point F>
struct common_arithmetic_type_impl<I, F> {
    // when one integral and one floating-point type are provided,
    // use the floating-point type
    using type = F;
};

template <std::floating_point F, std::integral I>
struct common_arithmetic_type_impl<F, I> : common_arithmetic_type_impl<I, F> {};

template <std::floating_point F1, std::floating_point F2>
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

////////////////////////////////////////////////////////////////
/////////////// common_arithmetic_type_filtered_t///////////////
////////////////////////////////////////////////////////////////

namespace detail {

template <typename Tuple> struct common_arithmetic_type_from_tuple_impl;

template <typename... Ts>
    requires(sizeof...(Ts) > 0 &&
             (std::is_arithmetic_v<std::remove_cvref_t<Ts>> && ...))
struct common_arithmetic_type_from_tuple_impl<std::tuple<Ts...>>
    : common_arithmetic_type_impl<Ts...> {};

template <typename... Ts> struct nullopt_filter_impl {
    // no type defined
};

template <> struct nullopt_filter_impl<> {
    using type = std::tuple<>;
};

template <typename T, typename... Ts> struct nullopt_filter_impl<T, Ts...> {
  private:
    using arg_t = T;
    using tail_t = typename nullopt_filter_impl<Ts...>::type;

  public:
    using type = std::conditional_t<is_nullopt_t_v<T>, tail_t,
                                    decltype(std::tuple_cat(
                                        std::declval<std::tuple<arg_t>>(),
                                        std::declval<tail_t>()))>;
};

template <typename T>
constexpr bool is_arithmetic_or_nullopt_v =
    std::is_arithmetic_v<T> || is_nullopt_t_v<T>;

} // namespace detail

template <typename... Ts>
    requires(sizeof...(Ts) > 0 &&
             (detail::is_arithmetic_or_nullopt_v<std::remove_cvref_t<Ts>> &&
              ...))
using filtered_common_arithmetic_type_t =
    typename detail::common_arithmetic_type_from_tuple_impl<
        typename detail::nullopt_filter_impl<
            std::remove_cvref_t<Ts>...>::type>::type;

} // namespace mdtensor::core
