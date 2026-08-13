/**
 * @file
 * @brief Minimum reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../logic/all.hpp"
#include "../util/fill.hpp"
#include "minimum.hpp"

namespace mdtensor {

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO, std::integral axes_t,
          axes_t... axes, typename out_t = std::nullopt_t,
          typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto min(auto &&in,
                                 std::integer_sequence<axes_t, axes...>,
                                 out_t &&out = out_t{std::nullopt},
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using calc_t = core::calc_type_t<dtype, decltype(in_mds)>;

    auto out_md = core::resolve_reduced_output<calc_t, keepdims>(
        std::forward<decltype(out)>(out),
        std::integer_sequence<axes_t, axes...>{}, core::extents<std::uint8_t>{},
        in_mds);

    auto init = full_like<bool, backend>(out_md, false);

    if constexpr (!core::nullopt_t_c<decltype(initial)>) {
        static_cast<void>(full_like<void>(
            out_md, std::forward<decltype(initial)>(initial), out_md));
        fill(init, true);
    }

    core::reduce<keepdims>(
        [&](auto &&...elems) {
            core::batch<backend>(
                [](auto &&in_u, auto &&out_u, auto &&init_u, auto &&where_u) {
                    if (core::initialize_ufunc(
                            std::forward<decltype(init_u)>(init_u),
                            std::forward<decltype(where_u)>(where_u))) {
                        ufunc::copy_ufunc(std::forward<decltype(in_u)>(in_u),
                                          std::forward<decltype(out_u)>(out_u));

                    } else {
                        ufunc::minimum_ufunc(
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
            "mdtensor::min: cannot initialize output tensor.");
    }

    return out_md;
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
    requires(!core::integer_sequence_c<initial_t>)
[[nodiscard]] constexpr auto min(auto &&in,
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return min<dtype, false, backend>(std::forward<decltype(in)>(in),
                                      std::index_sequence<>{}, std::nullopt,
                                      std::forward<decltype(initial)>(initial),
                                      std::forward<decltype(where)>(where));
}

} // namespace mdtensor
