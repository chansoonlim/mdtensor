/**
 * @file
 * @brief Logical all-reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../util/fill.hpp"
#include "logical_and.hpp"

namespace mdtensor {
namespace rfunc {

template <core::Backend backend>
constexpr void all_rfunc(auto &&in, auto &&out, auto &&where) {
    const auto out_mds = core::to_mdspan(std::forward<decltype(out)>(out));

    const auto mask = [&]() {
        if constexpr (core::nullopt_t_value_type_c<decltype(where)>) {
            return out_mds;

        } else {
            return logical_and<bool, backend>(
                out_mds, std::forward<decltype(where)>(where));
        }
    }();

    static_cast<void>(logical_and<void, backend>(std::forward<decltype(in)>(in),
                                                 out_mds, out_mds, mask));
}

} // namespace rfunc

template <typename dtype = bool, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO, std::integral axes_t,
          axes_t... axes, typename out_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto all(auto &&in,
                                 std::integer_sequence<axes_t, axes...>,
                                 out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = core::resolve_reduced_output<dtype, keepdims>(
        std::forward<decltype(out)>(out),
        std::integer_sequence<axes_t, axes...>{}, core::extents<std::uint8_t>{},
        in_mds);

    // TODO: move to reduce
    fill<backend>(out_md, true);

    // TODO: use escape for each ufunc to remove mask calculation overhead
    core::reduce<keepdims>(
        [&](auto &&...elems) {
            rfunc::all_rfunc<backend>(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<axes_t, axes...>{},
        std::index_sequence<0, 0, 0>{},
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

template <std::int64_t axis, typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto all(auto &&in, out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return all<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in),
        std::integer_sequence<std::int64_t, axis>{},
        std::forward<decltype(out)>(out), std::forward<decltype(where)>(where));
}

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto all(auto &&in, out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return all<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in), std::index_sequence<>{},
        std::forward<decltype(out)>(out), std::forward<decltype(where)>(where));
}

} // namespace mdtensor
