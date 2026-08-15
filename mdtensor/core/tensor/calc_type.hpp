/**
 * @file
 * @brief Calculation type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../ufunc/common_value_type.hpp"

namespace mdtensor::core {
namespace detail {

template <typename dtype, typename... Ts> struct calc_type_impl;

template <typename dtype, typename... Ts>
    requires(!std::is_void_v<std::remove_cvref_t<dtype>>)
struct calc_type_impl<dtype, Ts...> {
    using type = std::remove_cvref_t<dtype>;
};

template <typename... Ts>
    requires(sizeof...(Ts) > 0)
struct calc_type_impl<void, Ts...> {
    using type = core::common_value_type_t<Ts...>;
};

} // namespace detail

template <typename dtype, typename... Ts>
using calc_type_t = detail::calc_type_impl<dtype, Ts...>::type;

template <typename dtype, typename... Ts>
using signed_calc_type_t =
    promote_type_t<calc_type_t<dtype, Ts...>, std::int8_t>;

template <typename dtype, typename... Ts>
using floating_calc_type_t = promote_type_t<calc_type_t<dtype, Ts...>, float>;

} // namespace mdtensor::core
