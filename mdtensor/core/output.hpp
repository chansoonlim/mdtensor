/**
 * @file
 * @brief Output utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "broadcast.hpp"
#include "container.hpp"

namespace mdtensor::core {
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
        return make_container<dtype>(std::forward<uout_exts_t>(uout_exts));

    } else {
        using value_t = output_value_t<dtype, decltype(ins)...>;

        // calculate broadcasted extents
        const auto bexts = detail::get_broadcast_extents(
            std::index_sequence<uranks...>{},
            to_const_mdspan(std::forward<decltype(ins)>(ins))...);

        // make output container
        return make_container<value_t>(
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

template <typename dtype = void, std::size_t... uranks,
          extents_tuple_c uout_exts_tuple_t>
[[nodiscard]] constexpr auto make_outputs(std::index_sequence<uranks...>,
                                          uout_exts_tuple_t &&uout_exts_tuple,
                                          auto &&...ins) {
    static_assert(sizeof...(uranks) == sizeof...(ins),
                  "Number of uranks must match number of inputs.");

    constexpr std::size_t ins_num = sizeof...(uranks);
    constexpr std::size_t outs_num =
        std::tuple_size_v<std::remove_cvref_t<uout_exts_tuple_t>>;

    if constexpr (ins_num == 0) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::tuple{
                make_container<dtype>(std::get<Is>(uout_exts_tuple))...};
        }(std::make_index_sequence<outs_num>{});

    } else {
        using value_t = output_value_t<dtype, decltype(ins)...>;

        // calculate broadcasted extents
        const auto bexts = detail::get_broadcast_extents(
            std::index_sequence<uranks...>{},
            to_const_mdspan(std::forward<decltype(ins)>(ins))...);

        // make output container
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::tuple{make_container<value_t>(
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

} // namespace mdtensor::core
