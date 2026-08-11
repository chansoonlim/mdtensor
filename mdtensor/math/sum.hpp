/**
 * @file
 * @brief Sum reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../util/fill.hpp"
#include "add.hpp"

namespace mdtensor {

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO, std::integral axes_t,
          axes_t... axes, typename out_t = std::nullopt_t,
          typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto sum(auto &&in,
                                 std::integer_sequence<axes_t, axes...>,
                                 out_t &&out = out_t{std::nullopt},
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_reduce_output<dtype, keepdims>(
                std::integer_sequence<axes_t, axes...>{},
                std::index_sequence<0>{}, core::extents<std::uint8_t>{},
                in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    // TODO: move to reduce
    if constexpr (core::nullopt_t_c<decltype(initial)>) {
        fill<backend>(out_md, 0);

    } else {
        fill<backend>(out_md, std::forward<decltype(initial)>(initial));
    }

    core::reduce<keepdims>(
        [](auto &&in, auto &&out, auto &&where) {
            static_cast<void>(
                add<void, backend>(std::forward<decltype(in)>(in),
                                   std::forward<decltype(out)>(out),
                                   std::forward<decltype(out)>(out),
                                   std::forward<decltype(where)>(where)));
        },
        std::integer_sequence<axes_t, axes...>{},
        std::index_sequence<0, 0, 0>{},
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

template <std::int64_t axis, typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto sum(auto &&in, out_t &&out = out_t{std::nullopt},
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return sum<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in),
        std::integer_sequence<std::int64_t, axis>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial),
        std::forward<decltype(where)>(where));
}

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto sum(auto &&in, out_t &&out = out_t{std::nullopt},
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return sum<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in), std::index_sequence<>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial),
        std::forward<decltype(where)>(where));
}

} // namespace mdtensor
