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

template <std::floating_point dtype>
[[nodiscard]] constexpr dtype
sqrt_newton_raphson(const dtype &x, const dtype &curr, const dtype &prev) {
    return (curr == prev)
               ? curr
               : sqrt_newton_raphson(x, (curr + x / curr) / (dtype)2, curr);
}

constexpr void sqrt_ufunc_native(auto &&in, auto &&out) {
    using calc_t = core::common_data_type_t<decltype(in), float>;

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

    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(in)>, bool>) {
        out = static_cast<calc_t>(in);
        return;

    } else {
        out = (in >= 0 && in < std::numeric_limits<calc_t>::infinity())
                  ? sqrt_newton_raphson(static_cast<calc_t>(in),
                                        static_cast<calc_t>(in),
                                        static_cast<calc_t>(0))
                  : std::numeric_limits<calc_t>::quiet_NaN();
    }
}

constexpr void sqrt_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where == false } -> std::convertible_to<bool>;
                  }) {
        if (where == false) {
            return;
        }
    }

#ifdef REAL_GCC
    if (!std::is_constant_evaluated()) {
        if constexpr (requires { out = std::sqrt(in); }) {
            out = std::sqrt(in);
            return;
        }
    }

#endif

    sqrt_ufunc_native(std::forward<decltype(in)>(in),
                      std::forward<decltype(out)>(out));
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto sqrt(auto &&in, out_t &&out = out_t{std::nullopt},
                                  where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = core::resolve_output_like<dtype, true>(
        std::forward<decltype(out)>(out), in_mds);

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::sqrt_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
