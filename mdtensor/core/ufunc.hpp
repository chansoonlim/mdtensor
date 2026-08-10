/**
 * @file
 * @brief Universal function (ufunc) utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "broadcast.hpp"
#include "manipulation.hpp"
#include "output.hpp"

namespace mdtensor::core {
namespace detail {

template <bool is_input, std::size_t axis, bool keepdims, mdspan_c io_t,
          typename index_t>
[[nodiscard]] constexpr auto reduce_input(io_t &&io, index_t i) {
    if constexpr (is_input) {
        return submdspan_from_left<axis>(std::forward<io_t>(io), i);

    } else if constexpr (keepdims) {
        return submdspan_from_left<axis>(std::forward<io_t>(io), 0);

    } else {
        return std::forward<io_t>(io);
    }
}

template <bool keepdims, extents_c bext_t, bool... is_input, mdspan_c... ios_t>
constexpr void batch_reduced(auto &&func, bext_t &&, std::index_sequence<>,
                             std::integer_sequence<bool, is_input...>,
                             ios_t &&...ios) {
    func(std::forward<ios_t>(ios)...);
}

template <bool keepdims, extents_c bext_t, std::size_t axis,
          std::size_t... axes, bool... is_input, mdspan_c... ios_t>
constexpr void
batch_reduced(auto &&func, bext_t &&bext, std::index_sequence<axis, axes...>,
              std::integer_sequence<bool, is_input...>, ios_t &&...ios) {
    static_assert(sizeof...(is_input) == sizeof...(ios_t));
    static_assert(((axis > axes) && ...), "Axes must be in descending order.");

    using index_t = typename std::remove_cvref_t<bext_t>::index_type;

    for (index_t i = 0; i < bext.extent(axis); i++) {
        batch_reduced<keepdims>(std::forward<decltype(func)>(func),
                                std::forward<decltype(bext)>(bext),
                                std::index_sequence<axes...>{},
                                std::integer_sequence<bool, is_input...>{},
                                reduce_input<is_input, axis, keepdims>(
                                    std::forward<ios_t>(ios), i)...);
    }
}

template <std::size_t... uranks, bool... is_input>
[[nodiscard]] constexpr auto
broadcast_only_input(std::index_sequence<uranks...>,
                     std::integer_sequence<bool, is_input...>, auto &&...ios) {
    static_assert(sizeof...(uranks) == sizeof...(ios));
    static_assert(sizeof...(is_input) == sizeof...(ios));

    // make helpers
    constexpr auto bin = std::array{is_input...};
    const auto ios_org =
        std::tuple{to_mdspan(std::forward<decltype(ios)>(ios))...};
    constexpr auto ios_num = std::tuple_size_v<decltype(ios_org)>;
    constexpr auto ios_uranks = std::array{uranks...};

    // separate inputs
    const auto ins_tuple = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        const auto get_input = [&]<std::size_t I>() {
            if constexpr (bin[I]) {
                return std::forward_as_tuple(std::get<I>(ios_org));

            } else {
                return std::forward_as_tuple();
            }
        };

        return std::tuple_cat(get_input.template operator()<Is>()...);
    }(std::make_index_sequence<sizeof...(ios)>{});

    constexpr auto ins_num = std::tuple_size_v<decltype(ins_tuple)>;

    static_assert(ins_num > 0, "At least one input must be provided.");

    constexpr auto ins_uranks = [&]() {
        auto ins_uranks = std::array<std::size_t, ins_num>{};
        std::size_t ins_idx = 0;
        for (std::size_t i = 0; i < ios_num; i++) {
            if (bin[i]) {
                ins_uranks[ins_idx++] = ios_uranks[i];
            }
        }
        return ins_uranks;
    }();

    // broadcast inputs only
    const auto [ins_bcast, ins_bexts] =
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return broadcast(std::index_sequence<ins_uranks[Is]...>{},
                             std::integer_sequence<bool, (void(Is), true)...>{},
                             std::get<Is>(ins_tuple)...);
        }(std::make_index_sequence<ins_num>{});

    // return broadcasted inputs and outputs in same order as original inputs
    return std::make_tuple(
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            const auto get_io = [&]<std::size_t I>() {
                constexpr std::size_t J =
                    [&]<std::size_t... Js>(std::index_sequence<Js...>) {
                        return (std::size_t{0} + ... +
                                static_cast<std::size_t>(bin[Js]));
                    }(std::make_index_sequence<I>{});

                if constexpr (bin[I]) {
                    // pass broadcasted input
                    return std::get<J>(ins_bcast);

                } else {
                    // pass through outputs
                    return std::get<I>(ios_org);
                }
            };

            return std::make_tuple(get_io.template operator()<Is>()...);
        }(std::make_index_sequence<sizeof...(ios)>{}),
        ins_bexts);
}

} // namespace detail

template <bool keepdims = false, std::integral axes_t, axes_t... axes,
          std::size_t... uranks, bool... is_input>
constexpr void reduce(auto &&func, std::integer_sequence<axes_t, axes...>,
                      std::index_sequence<uranks...>,
                      std::integer_sequence<bool, is_input...>, auto &&...ios) {
    // broadcast inputs only
    const auto [ios_bcast, ins_bexts] =
        detail::broadcast_only_input(std::index_sequence<uranks...>{},
                                     std::integer_sequence<bool, is_input...>{},
                                     std::forward<decltype(ios)>(ios)...);

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

    // batch
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        [&]<std::size_t... Js>(std::index_sequence<Js...>) {
            detail::batch_reduced<keepdims>(
                std::forward<decltype(func)>(func), ins_bexts,
                std::index_sequence<axes_sorted[Js]...>{},
                std::integer_sequence<bool, is_input...>{},
                std::get<Is>(ios_bcast)...);
        }(std::make_index_sequence<axes_sorted.size()>{});
    }(std::make_index_sequence<sizeof...(ios)>{});
}

template <typename dtype = void, bool keepdims = false, std::integral axes_t,
          axes_t... axes, std::size_t... uranks, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
make_reduce_output(std::integer_sequence<axes_t, axes...>,
                   std::index_sequence<uranks...>, uout_exts_t &&uout_exts,
                   auto &&...ins) {
    static_assert(sizeof...(ins) > 0, "At least one input must be provided.");
    static_assert(sizeof...(uranks) == sizeof...(ins),
                  "Number of uranks must match number of inputs.");

    // calculate input broadcasted extents
    const auto ins_bexts = detail::get_broadcast_extents(
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
                               (contains(axes_sorted, Is)
                                    ? std::size_t{1}
                                    : ins_bexts_t::static_extent(Is))...>{
                    (contains(axes_sorted, Is)
                         ? index_t{1}
                         : static_cast<index_t>(ins_bexts.extent(Is)))...};
            }(std::make_index_sequence<ins_bexts_t::rank()>{});

        } else {
            // generate unselected axes sequence
            constexpr auto not_axes_arr = [&]() {
                auto not_axes_arr = std::array<std::size_t, not_axes_size>{};

                std::size_t not_axes_idx = 0;
                for (std::size_t i = 0; i < ins_bexts_t::rank(); i++) {
                    if (!contains(axes_sorted, i)) {
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

    return make_container<value_t>(
        compose_extents(out_bexts, std::forward<uout_exts_t>(uout_exts)));
}

// TODO: develop make_reduce_outputs

} // namespace mdtensor::core
