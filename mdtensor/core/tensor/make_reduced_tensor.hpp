/**
 * @file
 * @brief Output utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../ufunc/ufunc.hpp"

namespace mdtensor::core {
namespace detail {

template <typename value_t, std::size_t size>
[[nodiscard]] constexpr bool contains(const std::array<value_t, size> &array,
                                      const value_t &value) noexcept {
    for (const auto element : array) {
        if (element == value) {
            return true;
        }
    }

    return false;
}

} // namespace detail

template <bool keepdims = false, std::integral axes_t, axes_t... axes,
          std::size_t... uranks, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
make_reduced_extents(std::integer_sequence<axes_t, axes...>,
                     std::index_sequence<uranks...>, uout_exts_t &&uout_exts,
                     auto &&...ins) {
    static_assert(sizeof...(ins) > 0, "At least one input must be provided.");
    static_assert(sizeof...(uranks) == sizeof...(ins),
                  "Number of uranks must match number of inputs.");

    // calculate input broadcasted extents
    const auto ins_bexts = get_broadcast_extents(
        std::index_sequence<uranks...>{},
        to_const_mdspan(std::forward<decltype(ins)>(ins))...);

    using ins_bexts_t = decltype(ins_bexts);

    // get sorted array
    constexpr auto axes_sorted = [&]() {
        if constexpr (ins_bexts_t::rank() == 0) {
            return std::array<std::size_t, 0>{};

        } else if constexpr (sizeof...(axes) == 0) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::array{(ins_bexts_t::rank() - 1 - Is)...};
            }(std::make_index_sequence<ins_bexts_t::rank()>{});

        } else {
            return get_sorted_axes<ins_bexts_t::rank()>(
                std::integer_sequence<axes_t, axes...>{},
                std::greater<std::size_t>{});
        }
    }();

    // generate out_bexts
    constexpr auto not_axes_size = ins_bexts_t::rank() - axes_sorted.size();

    const auto out_bexts = [&]() {
        using index_t = typename ins_bexts_t::index_type;

        if constexpr (keepdims) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return extents<index_t,
                               (detail::contains(axes_sorted, Is)
                                    ? std::size_t{1}
                                    : ins_bexts_t::static_extent(Is))...>{
                    (detail::contains(axes_sorted, Is)
                         ? index_t{1}
                         : static_cast<index_t>(ins_bexts.extent(Is)))...};
            }(std::make_index_sequence<ins_bexts_t::rank()>{});

        } else {
            // generate unselected axes sequence
            constexpr auto not_axes_arr = [&]() {
                auto not_axes_arr = std::array<std::size_t, not_axes_size>{};

                std::size_t not_axes_idx = 0;
                for (std::size_t i = 0; i < ins_bexts_t::rank(); i++) {
                    if (!detail::contains(axes_sorted, i)) {
                        not_axes_arr[not_axes_idx++] = i;
                    }
                }

                return not_axes_arr;
            }();

            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return extents<index_t,
                               ins_bexts_t::static_extent(not_axes_arr[Is])...>{
                    ins_bexts.extent(not_axes_arr[Is])...};
            }(std::make_index_sequence<not_axes_arr.size()>{});
        }
    }();

    return compose_extents(out_bexts, std::forward<uout_exts_t>(uout_exts));
}

template <typename dtype = void, bool keepdims = false, std::integral axes_t,
          axes_t... axes, std::size_t... uranks, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
make_reduced_tensor(std::integer_sequence<axes_t, axes...>,
                    std::index_sequence<uranks...>, uout_exts_t &&uout_exts,
                    auto &&...ins) {
    using value_t = calc_type_t<dtype, decltype(ins)...>;

    const auto out_exts = make_reduced_extents<keepdims>(
        std::integer_sequence<axes_t, axes...>{},
        std::index_sequence<uranks...>{}, std::forward<uout_exts_t>(uout_exts),
        std::forward<decltype(ins)>(ins)...);

    return make_tensor<value_t>(out_exts);
}

template <typename dtype = void, bool keepdims = false, std::integral axes_t,
          axes_t... axes, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
make_reduced_tensor(std::integer_sequence<axes_t, axes...>,
                    uout_exts_t &&uout_exts, auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return make_reduced_tensor<dtype, keepdims>(
            std::integer_sequence<axes_t, axes...>{},
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

template <typename dtype = void, bool keepdims = false, std::integral axes_t,
          axes_t... axes, std::size_t... uranks>
[[nodiscard]] constexpr auto
make_reduced_tensors(std::integer_sequence<axes_t, axes...>,
                     std::index_sequence<uranks...>, auto &&uout_exts_tuple,
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
        using value_t = calc_type_t<dtype, decltype(ins)...>;

        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::tuple{
                make_tensor<value_t>(make_reduced_extents<keepdims>(
                    std::integer_sequence<axes_t, axes...>{},
                    std::index_sequence<uranks...>{},
                    std::get<Is>(uout_exts_tuple),
                    std::forward<decltype(ins)>(ins)...))...};
        }(std::make_index_sequence<outs_num>{});
    }
}

template <typename dtype = void, bool keepdims = false, std::integral axes_t,
          axes_t... axes>
[[nodiscard]] constexpr auto
make_reduced_tensors(std::integer_sequence<axes_t, axes...>,
                     auto &&uout_exts_tuple, auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return make_reduced_tensors<dtype, keepdims>(
            std::integer_sequence<axes_t, axes...>{},
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<decltype(uout_exts_tuple)>(uout_exts_tuple),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

} // namespace mdtensor::core
