/**
 * @file
 * @brief Make tensor utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "make_tensor.hpp"

namespace mdtensor::core {

template <typename T>
using to_mdspan_t = decltype(to_mdspan(std::declval<T>()));

template <typename... Ts>
using common_value_type_t = core::common_arithmetic_type_t<
    typename std::remove_cvref_t<to_mdspan_t<Ts>>::value_type...>;

namespace detail {

template <typename dtype, typename... Ts> struct output_value {
    using type = dtype;
};

template <typename... Ts> struct output_value<void, Ts...> {
    using type = core::common_arithmetic_type_t<
        typename std::remove_cvref_t<to_mdspan_t<Ts>>::value_type...>;
};

template <typename dtype, typename Tuple> struct output_value_from_tuple;

template <typename dtype, typename... Ts>
struct output_value_from_tuple<dtype, std::tuple<Ts...>>
    : output_value<dtype, Ts...> {};
} // namespace detail

template <typename dtype, typename... Ts>
using output_value_t = typename detail::output_value<dtype, Ts...>::type;

namespace detail {

template <typename T> struct unwrap_optional {
    using type = std::remove_cvref_t<T>;
};

template <typename T> struct unwrap_optional<std::optional<T>> {
    using type = std::remove_cvref_t<T>;
};

template <typename... Ts> struct filter_nullopt;

template <> struct filter_nullopt<> {
    using type = std::tuple<>;
};

template <typename T, typename... Ts> struct filter_nullopt<T, Ts...> {
  private:
    using arg_t = typename unwrap_optional<std::remove_cvref_t<T>>::type;
    using tail_t = typename filter_nullopt<Ts...>::type;

  public:
    using type = std::conditional_t<
        std::same_as<std::remove_cvref_t<T>, std::nullopt_t>, tail_t,
        decltype(std::tuple_cat(std::declval<std::tuple<arg_t>>(),
                                std::declval<tail_t>()))>;
};

} // namespace detail

template <typename dtype, typename... Ts>
using output_value_with_nullopt_t = typename detail::output_value_from_tuple<
    dtype, typename detail::filter_nullopt<Ts...>::type>::type;

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

namespace new_detail {

//////////////////////////////////////////////////////////////////////
///////// unwrap_optional ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

template <typename T> struct unwrap_optional {
    using type = std::remove_cvref_t<T>;
};

template <typename T> struct unwrap_optional<std::optional<T>> {
    using type = std::remove_cvref_t<T>;
};

//////////////////////////////////////////////////////////////////////
///////// unwrap_mdtype //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

// template <typename T>
// using to_mdspan_t = decltype(to_mdspan(std::declval<T>()));

template <typename T> struct unwrap_mdtype {
    using value_type = std::remove_cvref_t<T>;
};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename AccessorType>
struct unwrap_mdtype<
    mdspan<ElementType, ExtentsType, LayoutType, AccessorType>> {
    using value_type = typename std::remove_cvref_t<
        mdspan<ElementType, ExtentsType, LayoutType, AccessorType>>::value_type;
};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename ContainerType>
struct unwrap_mdtype<
    mdarray<ElementType, ExtentsType, LayoutType, ContainerType>> {
    using value_type = typename std::remove_cvref_t<mdarray<
        ElementType, ExtentsType, LayoutType, ContainerType>>::value_type;
};

//////////////////////////////////////////////////////////////////////
///////// unwrap_all /////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

template <typename T> struct unwrap_all {
    using type = std::remove_cvref_t<T>;
};

template <typename T> struct unwrap_all<std::optional<T>> {
    using type = unwrap_all<typename unwrap_optional<T>::type>::type;
};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename AccessorType>
struct unwrap_all<mdspan<ElementType, ExtentsType, LayoutType, AccessorType>> {
    using type = unwrap_all<typename unwrap_mdtype<mdspan<
        ElementType, ExtentsType, LayoutType, AccessorType>>::value_type>::type;
};

//////////////////////////////////////////////////////////////////////
/////////////// filter ///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

template <typename... Ts> struct value_type_filter {
    // no type defined
};

template <> struct value_type_filter<> {
    using type = std::tuple<>;
};

template <typename T, typename... Ts> struct value_type_filter<T, Ts...> {
  private:
    using arg_t = typename unwrap_all<T>::type;
    using tail_t = typename value_type_filter<Ts...>::type;

  public:
    using type = std::conditional_t<nullopt_t_c<arg_t>, tail_t,
                                    decltype(std::tuple_cat(
                                        std::declval<std::tuple<arg_t>>(),
                                        std::declval<tail_t>()))>;
};

//////////////////////////////////////////////////////////////////////
/////////////// output_value_from_tuple //////////////////////////////
//////////////////////////////////////////////////////////////////////

template <typename dtype, typename... Ts> struct choose_dtype_or {
    using type = dtype;
};

template <typename... Ts>
    requires(sizeof...(Ts) > 0 &&
             (std::is_arithmetic_v<std::remove_cvref_t<Ts>> && ...))
struct choose_dtype_or<void, Ts...> {
    using type = core::common_arithmetic_type_t<Ts...>;
};

template <typename dtype, typename Tuple> struct get_value_from_filter;

template <typename dtype, typename... Ts>
struct get_value_from_filter<dtype, std::tuple<Ts...>>
    : choose_dtype_or<dtype, Ts...> {};

} // namespace new_detail

//////////////////////////////////////////////////////////////////////
/////////////// output_value_type_t //////////////////////////////////
//////////////////////////////////////////////////////////////////////

template <typename dtype, typename... Ts>
using output_value_type_t = typename new_detail::get_value_from_filter<
    dtype, typename new_detail::value_type_filter<Ts...>::type>::type;

} // namespace mdtensor::core
