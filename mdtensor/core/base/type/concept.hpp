/**
 * @file
 * @brief Concepts
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <concepts>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "kokkos.hpp"

namespace mdtensor::core {

///////////////////////////////////////////////////////////////////////
///////// arithmetic concepts /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

template <typename T>
concept bool_c = std::is_same_v<std::remove_cvref_t<T>, bool>;

template <typename T>
concept integral_c = std::is_integral_v<std::remove_cvref_t<T>>;

template <typename T>
concept floating_point_c = std::is_floating_point_v<std::remove_cvref_t<T>>;

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

template <typename T>
concept arithmetic_c = arithmetic<std::remove_cvref_t<T>>;

///////////////////////////////////////////////////////////////////////
///////// nullopt_t concept ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

namespace detail {

template <typename T> struct is_nullopt_impl : std::false_type {};

template <> struct is_nullopt_impl<std::nullopt_t> : std::true_type {};

template <typename T> struct is_nullopt_t : is_nullopt_impl<T> {};

template <typename T> constexpr bool is_nullopt_t_v = is_nullopt_t<T>::value;

} // namespace detail

template <typename T>
concept nullopt_t_c = detail::is_nullopt_t_v<std::remove_cvref_t<T>>;

///////////////////////////////////////////////////////////////////////
///////// integer_sequence concept ////////////////////////////////////
///////////////////////////////////////////////////////////////////////

namespace detail {

template <typename T> struct is_integer_sequence_impl : std::false_type {};

template <typename T, T... Ints>
struct is_integer_sequence_impl<std::integer_sequence<T, Ints...>>
    : std::true_type {};

template <typename T>
struct is_integer_sequence : is_integer_sequence_impl<T> {};

template <typename T>
constexpr bool is_integer_sequence_v = is_integer_sequence<T>::value;

} // namespace detail

template <typename T>
concept integer_sequence_c =
    detail::is_integer_sequence_v<std::remove_cvref_t<T>>;

///////////////////////////////////////////////////////////////////////
///////// extents concept /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

namespace detail {

template <typename T> struct is_extents_impl : std::false_type {};

template <typename IndexType, std::size_t... Extents>
struct is_extents_impl<extents<IndexType, Extents...>> : std::true_type {};

// NOTE: stdex::detail::__is_extents is not used for godbolt test compatibility
template <typename T> struct is_extents : is_extents_impl<T> {};

template <typename T> constexpr bool is_extents_v = is_extents<T>::value;

} // namespace detail

template <typename T>
concept extents_c = detail::is_extents_v<std::remove_cvref_t<T>>;

///////////////////////////////////////////////////////////////////////
///////// mdspan concept //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

namespace detail {

template <typename T> struct is_mdspan_impl : std::false_type {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename AccessorType>
struct is_mdspan_impl<
    mdspan<ElementType, ExtentsType, LayoutType, AccessorType>>
    : std::true_type {};

template <typename T> struct is_mdspan : is_mdspan_impl<T> {};

template <typename T> constexpr bool is_mdspan_v = is_mdspan<T>::value;

} // namespace detail

template <typename T>
concept mdspan_c = detail::is_mdspan_v<std::remove_cvref_t<T>>;

///////////////////////////////////////////////////////////////////////
///////// mdarray concept //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

namespace detail {

template <typename T> struct is_mdarray_impl : std::false_type {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename ContainerType>
struct is_mdarray_impl<
    mdarray<ElementType, ExtentsType, LayoutType, ContainerType>>
    : std::true_type {};

template <typename T> struct is_mdarray : is_mdarray_impl<T> {};

template <typename T> constexpr bool is_mdarray_v = is_mdarray<T>::value;

} // namespace detail

template <typename T>
concept mdarray_c = detail::is_mdarray_v<std::remove_cvref_t<T>>;

} // namespace mdtensor::core
