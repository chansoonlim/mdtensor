/**
 * @file
 * @brief Exact array equality utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"
#include "isnan.hpp"

namespace mdtensor {
namespace ufunc {

template <bool equal_nan>
constexpr bool array_equal_ufunc(auto &&in1, auto &&in2) {
    if constexpr (equal_nan) {
        if (isnan(in1) && isnan(in2)) {
            return true;
        }
    }

    return in1 == in2;
}

} // namespace ufunc

[[nodiscard]] constexpr bool array_equal(auto &&in1, auto &&in2,
                                         const bool equal_nan = false) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    if constexpr (core::is_always_different_extents<
                      decltype(in1_mds.extents()),
                      decltype(in2_mds.extents())>()) {
        return false;

    } else {
        if (!core::is_same_extents(in1_mds.extents(), in2_mds.extents())) {
            return false;
        }

        const auto run_batch = [&]<bool equal_nan_v>() {
            return core::batch_while(
                [](auto &&...elems) {
                    return ufunc::array_equal_ufunc<equal_nan_v>(
                        std::forward<decltype(elems)>(elems)...);
                },
                in1_mds, in2_mds);
        };

        if (equal_nan) {
            return run_batch.template operator()<true>();

        } else {
            return run_batch.template operator()<false>();
        }
    }
}

} // namespace mdtensor
