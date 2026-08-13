/**
 * @file
 * @brief Logical any-reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "all.hpp"
#include "logical_or.hpp"

namespace mdtensor {

template <bool keepdims = false, core::Backend backend = core::Backend::AUTO,
          std::integral axes_t, axes_t... axes, typename out_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto any(auto &&in,
                                 std::integer_sequence<axes_t, axes...>,
                                 out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    // TODO: move batch outside of reduce, add escape when out is initialized
    // and already false. and specialization

    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto where_mds =
        core::to_const_mdspan(std::forward<decltype(where)>(where));

    if constexpr (sizeof...(axes) == 0) {
        bool init = false;

        const bool is_all_false = core::batch_while(
            [&](auto &&in_u, auto &&where_u) {
                static_cast<void>(core::initialize_ufunc(init, where_u));

                if constexpr (requires { static_cast<bool>(where_u); }) {
                    if (!static_cast<bool>(where_u)) {
                        return true;
                    }
                }

                return !static_cast<bool>(in_u);
            },
            std::integer_sequence<bool, true, true>{}, in_mds, where_mds);

        if (!init) {
            throw std::runtime_error(
                "mdtensor::any: cannot initialize output tensor.");
        }

        return !is_all_false;

    } else {
        auto out_md = core::resolve_reduced_output<bool, keepdims>(
            std::forward<decltype(out)>(out),
            std::integer_sequence<axes_t, axes...>{},
            core::extents<std::uint8_t>{}, in_mds);

        auto init = full_like<bool>(out_md, false);

        core::reduce<keepdims>(
            [&](auto &&...elems) {
                core::batch<backend>(
                    [](auto &&in_u, auto &&out_u, auto &&init_u,
                       auto &&where_u) {
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
            std::integer_sequence<bool, true, false, false, true>{}, in_mds,
            out_md, init, where_mds);

        if (!all(init)) {
            throw std::runtime_error(
                "mdtensor::any: cannot initialize output tensor.");
        }

        return out_md;
    }
}

template <core::Backend backend = core::Backend::AUTO,
          typename where_t = std::nullopt_t>
    requires(!core::integer_sequence_c<where_t>)
[[nodiscard]] constexpr bool any(auto &&in,
                                 where_t &&where = where_t{std::nullopt}) {
    return any<false, backend>(std::forward<decltype(in)>(in),
                               std::index_sequence<>{}, std::nullopt,
                               std::forward<decltype(where)>(where));
}

} // namespace mdtensor
