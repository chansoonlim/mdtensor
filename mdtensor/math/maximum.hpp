/**
 * @file
 * @brief Element-wise maximum utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace ufunc {

constexpr void maximum_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    using value_t = std::remove_cvref_t<decltype(out())>;

    // if one of the inputs is NaN, return NaN (numpy-like)
    if constexpr (requires {
                      { std::isnan(in1()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in1())) {
            out() = std::numeric_limits<value_t>::quiet_NaN();
            return;
        }
    }

    if constexpr (requires {
                      { std::isnan(in2()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in2())) {
            out() = std::numeric_limits<value_t>::quiet_NaN();
            return;
        }
    }

    out() = std::max(static_cast<value_t>(in1()), static_cast<value_t>(in2()));
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto maximum(auto &&in1, auto &&in2,
                                     out_t &&out = out_t{std::nullopt},
                                     where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::is_nullopt_t_c<decltype(out)>) {
            return core::make_output<dtype>(core::extents<std::uint8_t>{},
                                            in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::maximum_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{},
        std::forward<decltype(in1)>(in1), std::forward<decltype(in2)>(in2),
        out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
