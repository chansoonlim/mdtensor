/**
 * @file
 * @brief Make tensor utilities for mdtensor.
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
/////////////// choose_dtype_or //////////////////////////////////////
//////////////////////////////////////////////////////////////////////

template <typename dtype, typename... Ts> struct choose_dtype_or;

template <typename dtype, typename... Ts>
    requires(!std::is_void_v<std::remove_cvref_t<dtype>>)
struct choose_dtype_or<dtype, Ts...> {
    using type = std::remove_cvref_t<dtype>;
};

template <typename... Ts>
    requires(sizeof...(Ts) > 0 &&
             (std::is_arithmetic_v<std::remove_cvref_t<Ts>> && ...))
struct choose_dtype_or<void, Ts...> {
    using type = core::promote_type_t<std::remove_cvref_t<Ts>...>;
};

//////////////////////////////////////////////////////////////////////
/////////////// get_value_from_filter ////////////////////////////////
//////////////////////////////////////////////////////////////////////

template <typename dtype, typename Tuple> struct get_value_from_filter {};

template <typename dtype, typename... Ts>
struct get_value_from_filter<dtype, std::tuple<Ts...>>
    : choose_dtype_or<dtype, Ts...> {};

} // namespace detail

//////////////////////////////////////////////////////////////////////
/////////////// calc_type_t //////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

template <typename dtype, typename... Ts>
using calc_type_t = typename detail::get_value_from_filter<
    dtype, typename detail::value_type_filter<Ts...>::type>::type;

template <typename dtype, typename... Ts>
using signed_calc_type_t =
    promote_type_t<calc_type_t<dtype, Ts...>, std::int8_t>;

template <typename dtype, typename... Ts>
using floating_calc_type_t = promote_type_t<calc_type_t<dtype, Ts...>, float>;

} // namespace mdtensor::core
