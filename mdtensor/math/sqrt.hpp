/**
 * @file
 * @brief Element-wise square root utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../creation/empty_like.hpp"

namespace mdtensor {
namespace ufunc {
namespace detail {

template <std::floating_point calc_t>
[[nodiscard]] constexpr calc_t
sqrt_newton_raphson(const calc_t &x, const calc_t &curr, const calc_t &prev) {
    return (curr == prev)
               ? curr
               : sqrt_newton_raphson(x, (curr + x / curr) / (calc_t)2, curr);
}

} // namespace detail

template <typename dtype = void>
constexpr void sqrt_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires { static_cast<bool>(where); }) {
        if (!static_cast<bool>(where)) {
            return;
        }
    }

    if constexpr (requires {
                      { std::isnan(in) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in)) {
            out = in;
            return;
        }
    }

    if constexpr (requires {
                      { std::isinf(in) } -> std::convertible_to<bool>;
                  }) {
        if (std::isinf(in)) {
            out = in;
            return;
        }
    }

    if constexpr (core::bool_c<decltype(in)>) {
        out = in;
        return;

    } else {
        using calc_t =
            core::floating_calc_type_t<dtype, decltype(in), decltype(out)>;

        // NOTE: std::sqrt is not used here becase it is not constexpr in C++20.
        out = (in >= 0 && in < std::numeric_limits<calc_t>::infinity())
                  ? detail::sqrt_newton_raphson(static_cast<calc_t>(in),
                                                static_cast<calc_t>(in),
                                                static_cast<calc_t>(0))
                  : std::numeric_limits<calc_t>::quiet_NaN();
    }
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto sqrt(auto &&in, out_t &&out = out_t{std::nullopt},
                                  where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using calc_t = core::floating_calc_type_t<dtype, decltype(in_mds)>;

    auto out_md = core::resolve_output_like<calc_t>(
        std::forward<decltype(out)>(out), in_mds);

    core::batch<backend>(
        [](auto &&...elems) {
            ufunc::sqrt_ufunc<calc_t>(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
