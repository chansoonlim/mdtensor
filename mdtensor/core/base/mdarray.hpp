/**
 * @file
 * @brief Mdarray utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "type.hpp"

namespace mdtensor::core {

template <typename ElementType, typename Extents,
          typename LayoutPolicy = stdex::layout_right,
          typename Container = std::vector<ElementType>>
using mdarray = stdex::mdarray<ElementType, Extents, LayoutPolicy, Container>;

namespace detail {

template <typename T> struct is_mdarray_impl : std::false_type {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename ContainerType>
struct is_mdarray_impl<
    mdarray<ElementType, ExtentsType, LayoutType, ContainerType>>
    : std::true_type {};

} // namespace detail

template <typename T> struct is_mdarray : detail::is_mdarray_impl<T> {};

template <typename T> constexpr bool is_mdarray_v = is_mdarray<T>::value;

template <typename T>
concept mdarray_c = is_mdarray_v<std::remove_cvref_t<T>>;

} // namespace mdtensor::core
