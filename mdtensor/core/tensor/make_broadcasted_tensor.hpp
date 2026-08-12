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

template <typename dtype, typename... Ts> struct output_value {
    using type = dtype;
};

template <typename... Ts> struct output_value<void, Ts...> {
    using type = common_arithmetic_type_t<
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

template <typename dtype = void, std::size_t... uranks, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
make_broadcasted_tensor(std::index_sequence<uranks...>, uout_exts_t &&uout_exts,
                        auto &&...ins) {
    static_assert(sizeof...(uranks) == sizeof...(ins),
                  "Number of uranks must match number of inputs.");

    constexpr std::size_t ins_num = sizeof...(uranks);

    if constexpr (ins_num == 0) {
        return make_tensor<dtype>(std::forward<uout_exts_t>(uout_exts));

    } else {
        using value_t = output_value_t<dtype, decltype(ins)...>;

        // calculate broadcasted extents
        const auto bexts = get_broadcast_extents(
            std::index_sequence<uranks...>{},
            to_const_mdspan(std::forward<decltype(ins)>(ins))...);

        return make_tensor<value_t>(
            compose_extents(bexts, std::forward<uout_exts_t>(uout_exts)));
    }
}

template <typename dtype = void, extents_c uout_exts_t>
[[nodiscard]] constexpr auto make_broadcasted_tensor(uout_exts_t &&uout_exts,
                                                     auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return make_broadcasted_tensor<dtype>(
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

template <typename dtype = void, std::size_t... uranks>
[[nodiscard]] constexpr auto
make_broadcasted_tensors(std::index_sequence<uranks...>, auto &&uout_exts_tuple,
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
        const auto bexts = get_broadcast_extents(
            std::index_sequence<uranks...>{},
            to_const_mdspan(std::forward<decltype(ins)>(ins))...);

        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::tuple{make_tensor<value_t>(
                compose_extents(bexts, std::get<Is>(uout_exts_tuple)))...};
        }(std::make_index_sequence<outs_num>{});
    }
}

template <typename dtype = void>
[[nodiscard]] constexpr auto make_broadcasted_tensors(auto &&uout_exts_tuple,
                                                      auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return make_broadcasted_tensors<dtype>(
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<decltype(uout_exts_tuple)>(uout_exts_tuple),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

template <typename dtype = void, bool floating = false, std::size_t... uranks,
          extents_c uout_exts_t>
[[nodiscard]] constexpr auto
resolve_broadcasted_output(auto &&out, std::index_sequence<uranks...>,
                           uout_exts_t &&uout_exts, auto &&...ins) {
    if constexpr (floating) {
        // Ensure that the output type is at least float precision
        using value_t = core::output_value_t<dtype, float, decltype(ins)...>;

        const auto out_md = resolve_broadcasted_output<value_t, false>(
            std::forward<decltype(out)>(out), std::index_sequence<uranks...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);

        // Check that resolved output type is at least float precision
        static_assert(floating_point_c<
                          typename to_mdspan_t<decltype(out_md)>::value_type>,
                      "Resolved output type must be at least float precision.");

        return out_md;

    } else {
        if constexpr (nullopt_t_c<decltype(out)>) {
            return make_broadcasted_tensor<dtype>(
                std::index_sequence<uranks...>{},
                std::forward<uout_exts_t>(uout_exts),
                std::forward<decltype(ins)>(ins)...);

        } else {
            const auto out_mds =
                to_output_mdspan(std::forward<decltype(out)>(out));

            // TODO: check same extents with expected extents
            // TODO: check out_md type is enough precision for dtype, if dtype
            // is not void
            // TODO: use resolve_output.

            return out_mds;
        }
    }
}

template <typename dtype = void, bool floating = false, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
resolve_broadcasted_output(auto &&out, uout_exts_t &&uout_exts, auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return resolve_broadcasted_output<dtype, floating>(
            std::forward<decltype(out)>(out),
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

// TODO: develop resolve_broadcasted_outputs

} // namespace mdtensor::core
