/**
 * @file
 * @brief Make tensor utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "output_value_type.hpp"

namespace mdtensor::core {

template <typename value_t, extents_c exts_t>
[[nodiscard]] constexpr auto resolve_output(auto &&out, exts_t &&exts) {
    if constexpr (core::nullopt_t_c<decltype(out)>) {
        return make_tensor<value_t>(std::forward<exts_t>(exts));

    } else {
        const auto out_mds = to_output_mdspan(std::forward<decltype(out)>(out));

        using out_value_t = typename decltype(out_mds)::value_type;

        static_assert(
            std::same_as<core::promote_type_t<value_t, out_value_t>, //
                         out_value_t>,
            "Resolved output must not be less precise than desired.");

        static_assert(
            !is_always_different_extents<decltype(out_mds.extents()), exts_t>(),
            "Resolved output extents must match the provided extents.");

        if (!is_same_extents(out_mds.extents(), std::forward<exts_t>(exts))) {
            throw std::invalid_argument(
                "Resolved output extents must match the provided extents.");
        }

        return out_mds;
    }
}

template <typename dtype = void>
[[nodiscard]] constexpr auto resolve_output_like(auto &&out, auto &&in) {
    const auto in_mds = to_const_mdspan(std::forward<decltype(in)>(in));

    using value_t = core::calc_type_t<dtype, decltype(in_mds)>;

    return resolve_output<value_t>(std::forward<decltype(out)>(out),
                                   in_mds.extents());
}

} // namespace mdtensor::core
