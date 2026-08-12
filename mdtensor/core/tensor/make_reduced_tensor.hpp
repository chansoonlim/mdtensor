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
#include "tensor.hpp"

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

template <typename dtype = void, bool keepdims = false, std::integral axes_t,
          axes_t... axes, std::size_t... uranks, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
make_reduced_tensor(std::integer_sequence<axes_t, axes...>,
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

    // generate out
    using value_t = output_value_t<dtype, decltype(ins)...>;

    return make_tensor<value_t>(
        compose_extents(out_bexts, std::forward<uout_exts_t>(uout_exts)));
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

// TODO: develop make_reduce_outputs

template <typename dtype = void, bool keepdims = false, bool floating = false,
          std::integral axes_t, axes_t... axes, std::size_t... uranks,
          extents_c uout_exts_t>
[[nodiscard]] constexpr auto
resolve_reduced_output(auto &&out, std::integer_sequence<axes_t, axes...>,
                       std::index_sequence<uranks...>, uout_exts_t &&uout_exts,
                       auto &&...ins) {
    if constexpr (floating) {
        // Ensure that the output type is at least float precision
        using value_t = output_value_t<dtype, float, decltype(ins)...>;

        const auto out_md = resolve_reduced_output<value_t, keepdims, false>(
            std::forward<decltype(out)>(out),
            std::integer_sequence<axes_t, axes...>{},
            std::index_sequence<uranks...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);

        // Check that resolved output type is at least float precision
        static_assert(floating_point_c<
                          typename to_mdspan_t<decltype(out_md)>::value_type>,
                      "Resolved output type must be at least float precision.");

        return out_md;

    } else {
        if constexpr (nullopt_t_c<decltype(out)>) {
            return make_reduced_tensor<dtype, keepdims>(
                std::integer_sequence<axes_t, axes...>{},
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

template <typename dtype = void, bool keepdims = false, bool floating = false,
          std::integral axes_t, axes_t... axes, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
resolve_reduced_output(auto &&out, std::integer_sequence<axes_t, axes...>,
                       uout_exts_t &&uout_exts, auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return resolve_reduced_output<dtype, keepdims, floating>(
            std::forward<decltype(out)>(out),
            std::integer_sequence<axes_t, axes...>{},
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

// TODO: develop resolve_reduced_outputs

} // namespace mdtensor::core
