/**
 * @file
 * @brief Numpy-like common type promotion for md-like types (mdspan, mdarray,
 * etc.)
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../base/base.hpp"

namespace mdtensor::core {
namespace detail {

//////////////////////////////////////////////////////////////////////
///////// unwrap_all /////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

template <typename T> struct unwrap_all;

template <typename T> struct unwrap_all_impl {
    using type = T;
};

template <typename T>
struct unwrap_all_impl<std::optional<T>> : unwrap_all<T> {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename AccessorType>
struct unwrap_all_impl<
    mdspan<ElementType, ExtentsType, LayoutType, AccessorType>>
    : unwrap_all<typename mdspan<ElementType, ExtentsType, LayoutType,
                                 AccessorType>::value_type> {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename ContainerType>
struct unwrap_all_impl<
    mdarray<ElementType, ExtentsType, LayoutType, ContainerType>>
    : unwrap_all<typename mdarray<ElementType, ExtentsType, LayoutType,
                                  ContainerType>::value_type> {};

template <typename T>
struct unwrap_all : unwrap_all_impl<std::remove_cvref_t<T>> {};

//////////////////////////////////////////////////////////////////////
/////////////// value_type_filter ////////////////////////////////////
//////////////////////////////////////////////////////////////////////

template <typename... Ts> struct value_type_filter {};

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
/////////////// get_value_from_filter ////////////////////////////////
//////////////////////////////////////////////////////////////////////

template <typename Tuple> struct get_value_from_filter {};

template <typename... Ts> struct get_value_from_filter<std::tuple<Ts...>> {
    using type = core::promote_type_t<std::remove_cvref_t<Ts>...>;
};

} // namespace detail

//////////////////////////////////////////////////////////////////////
/////////////// common_value_type_t //////////////////////////////////
//////////////////////////////////////////////////////////////////////

template <typename... Ts>
using common_value_type_t = typename detail::get_value_from_filter<
    typename detail::value_type_filter<Ts...>::type>::type;

} // namespace mdtensor::core
