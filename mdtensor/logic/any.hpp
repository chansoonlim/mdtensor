/**
 * @file
 * @brief Logical any-reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../util/fill.hpp"
#include "all.hpp"
#include "logical_or.hpp"

namespace mdtensor {

template <typename dtype = bool, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO, std::integral axes_t,
          axes_t... axes, typename out_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto any(auto &&in,
                                 std::integer_sequence<axes_t, axes...>,
                                 out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = core::resolve_reduced_output<dtype, keepdims>(
        std::forward<decltype(out)>(out),
        std::integer_sequence<axes_t, axes...>{}, core::extents<std::uint8_t>{},
        in_mds);

    auto init = full_like<bool>(out_md, false);

    core::reduce<keepdims>(
        [&](auto &&...elems) {
            core::batch_with_broadcast<backend>(
                [](auto &&in_u, auto &&out_u, auto &&init_u, auto &&where_u) {
                    if (core::initialize_ufunc(
                            std::forward<decltype(init_u)>(init_u),
                            std::forward<decltype(where_u)>(where_u))) {
                        ufunc::copy_ufunc<bool>(
                            std::forward<decltype(in_u)>(in_u),
                            std::forward<decltype(out_u)>(out_u));

                    } else {
                        ufunc::logical_or_ufunc(
                            std::forward<decltype(in_u)>(in_u),
                            std::forward<decltype(out_u)>(out_u),
                            std::forward<decltype(out_u)>(out_u),
                            std::forward<decltype(where_u)>(where_u));
                    }
                },
                std::integer_sequence<bool, true, false, false, true>{},
                std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<axes_t, axes...>{},
        std::integer_sequence<bool, true, false, false, true>{}, in_mds, out_md,
        init, std::forward<decltype(where)>(where));

    if (!all(init)) {
        throw std::runtime_error(
            "mdtensor::any: cannot initialize output tensor.");
    }

    return out_md;
}

template <std::int64_t axis, typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto any(auto &&in, out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return any<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in),
        std::integer_sequence<std::int64_t, axis>{},
        std::forward<decltype(out)>(out), std::forward<decltype(where)>(where));
}

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto any(auto &&in, out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return any<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in), std::index_sequence<>{},
        std::forward<decltype(out)>(out), std::forward<decltype(where)>(where));
}

} // namespace mdtensor
