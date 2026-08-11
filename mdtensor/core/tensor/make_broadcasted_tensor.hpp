/**
 * @file
 * @brief Output utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../broadcast/broadcast.hpp"
#include "type.hpp"

namespace mdtensor::core {
namespace detail {

template <typename T>
constexpr bool data_bool_v = std::same_as<std::remove_cvref_t<T>, bool>;

template <typename T> constexpr bool data_fpoint_v = floating_point_c<T>;

template <typename T>
constexpr bool data_integer_v =
    std::integral<std::remove_cvref_t<T>> && !data_bool_v<T>;

template <typename T>
constexpr bool data_scalar_v =
    data_bool_v<T> || data_integer_v<T> || data_fpoint_v<T>;

template <typename T1, typename T2> struct common_data_pair_impl {
    // no type
};

// bool + bool -> bool
template <typename T1, typename T2>
    requires(data_bool_v<T1> && data_bool_v<T2>)
struct common_data_pair_impl<T1, T2> {
    using type = bool;
};

// bool + T -> T
template <typename B, typename T>
    requires(data_bool_v<B> && !data_bool_v<T> && data_scalar_v<T>)
struct common_data_pair_impl<B, T> {
    using type = std::remove_cvref_t<T>;
};

// T + bool -> T
template <typename T, typename B>
    requires(!data_bool_v<T> && data_scalar_v<T> && data_bool_v<B>)
struct common_data_pair_impl<T, B> {
    using type = std::remove_cvref_t<T>;
};

// floating + scalar -> floating
template <typename T1, typename T2>
    requires(!data_bool_v<T1> && !data_bool_v<T2> && data_scalar_v<T1> &&
             data_scalar_v<T2> && (data_fpoint_v<T1> || data_fpoint_v<T2>))
struct common_data_pair_impl<T1, T2> {
  private:
    using lhs_t = std::remove_cvref_t<T1>;
    using rhs_t = std::remove_cvref_t<T2>;

  public:
    using type = std::conditional_t<
        data_fpoint_v<T1> && data_fpoint_v<T2>,
        std::conditional_t<(sizeof(lhs_t) >= sizeof(rhs_t)), lhs_t, rhs_t>,
        std::conditional_t<data_fpoint_v<T1>, lhs_t, rhs_t>>;
};

// integer + integer -> common_index_type_t
template <typename T1, typename T2>
    requires(data_integer_v<T1> && data_integer_v<T2> &&
             requires {
                 typename common_index_type_t<std::remove_cvref_t<T1>,
                                              std::remove_cvref_t<T2>>;
             })
struct common_data_pair_impl<T1, T2> {
    using type =
        common_index_type_t<std::remove_cvref_t<T1>, std::remove_cvref_t<T2>>;
};

template <typename... Ts> struct data_promote_impl {
    // no type
};

template <typename T>
    requires data_scalar_v<T>
struct data_promote_impl<T> {
    using type = std::remove_cvref_t<T>;
};

template <typename T1, typename T2>
struct data_promote_impl<T1, T2>
    : common_data_pair_impl<std::remove_cvref_t<T1>, std::remove_cvref_t<T2>> {
};

template <typename T1, typename T2, typename... Ts>
    requires requires {
        typename common_data_pair_impl<std::remove_cvref_t<T1>,
                                       std::remove_cvref_t<T2>>::type;
    }
struct data_promote_impl<T1, T2, Ts...> {
  private:
    using pair_t =
        typename common_data_pair_impl<std::remove_cvref_t<T1>,
                                       std::remove_cvref_t<T2>>::type;

  public:
    using type = typename data_promote_impl<pair_t, Ts...>::type;
};

} // namespace detail

template <typename... Ts>
    requires(sizeof...(Ts) > 0 && (detail::data_scalar_v<Ts> && ...))
using common_data_type_t = typename detail::data_promote_impl<Ts...>::type;

namespace detail {

template <typename dtype, typename... Ts> struct output_value {
    using type = dtype;
};

template <typename... Ts> struct output_value<void, Ts...> {
    using type = common_data_type_t<
        typename std::remove_cvref_t<to_mdspan_t<Ts>>::value_type...>;
};

template <typename dtype, typename Tuple> struct output_value_from_tuple;

template <typename dtype, typename... Ts>
struct output_value_from_tuple<dtype, std::tuple<Ts...>>
    : output_value<dtype, Ts...> {};

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
using output_value_t = typename detail::output_value<dtype, Ts...>::type;

template <typename dtype, typename... Ts>
using output_value_with_nullopt_t = typename detail::output_value_from_tuple<
    dtype, typename detail::filter_nullopt<Ts...>::type>::type;

template <typename dtype = void, std::size_t... uranks, extents_c uout_exts_t>
[[nodiscard]] constexpr auto make_output(std::index_sequence<uranks...>,
                                         uout_exts_t &&uout_exts,
                                         auto &&...ins) {
    static_assert(sizeof...(uranks) == sizeof...(ins),
                  "Number of uranks must match number of inputs.");

    constexpr std::size_t ins_num = sizeof...(uranks);

    if constexpr (ins_num == 0) {
        return make_tensor<dtype>(std::forward<uout_exts_t>(uout_exts));

    } else {
        using value_t = output_value_t<dtype, decltype(ins)...>;

        // calculate broadcasted extents
        const auto bexts = detail::get_broadcast_extents(
            std::index_sequence<uranks...>{},
            to_const_mdspan(std::forward<decltype(ins)>(ins))...);

        return make_tensor<value_t>(
            compose_extents(bexts, std::forward<uout_exts_t>(uout_exts)));
    }
}

template <typename dtype = void, extents_c uout_exts_t>
[[nodiscard]] constexpr auto make_output(uout_exts_t &&uout_exts,
                                         auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return make_output<dtype>(std::index_sequence<((void)Is, 0)...>{},
                                  std::forward<uout_exts_t>(uout_exts),
                                  std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

template <typename dtype = void, std::size_t... uranks>
[[nodiscard]] constexpr auto make_outputs(std::index_sequence<uranks...>,
                                          auto &&uout_exts_tuple,
                                          auto &&...ins) {
    static_assert(sizeof...(uranks) == sizeof...(ins),
                  "Number of uranks must match number of inputs.");

    constexpr std::size_t ins_num = sizeof...(uranks);
    constexpr std::size_t outs_num =
        std::tuple_size_v<std::remove_cvref_t<decltype(uout_exts_tuple)>>;

    if constexpr (ins_num == 0) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::tuple{
                make_tensor<dtype>(std::get<Is>(uout_exts_tuple))...};
        }(std::make_index_sequence<outs_num>{});

    } else {
        using value_t = output_value_t<dtype, decltype(ins)...>;

        // calculate broadcasted extents
        const auto bexts = detail::get_broadcast_extents(
            std::index_sequence<uranks...>{},
            to_const_mdspan(std::forward<decltype(ins)>(ins))...);

        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::tuple{make_tensor<value_t>(
                compose_extents(bexts, std::get<Is>(uout_exts_tuple)))...};
        }(std::make_index_sequence<outs_num>{});
    }
}

template <typename dtype = void>
[[nodiscard]] constexpr auto make_outputs(auto &&uout_exts_tuple,
                                          auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return make_outputs<dtype>(
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<decltype(uout_exts_tuple)>(uout_exts_tuple),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

// TODO: develop make_reduce_outputs

} // namespace mdtensor::core
