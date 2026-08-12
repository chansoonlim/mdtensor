/**
 * @file
 * @brief Type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <tuple>
#include <type_traits>

#include "common_arithmetic_type.hpp"
#include "null.hpp"

namespace mdtensor::core {
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
