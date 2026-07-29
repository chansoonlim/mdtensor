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

namespace mdtensor {

template <typename in1_t, typename in2_t>
[[nodiscard]] inline constexpr bool array_equal(in1_t &&in1, in2_t &&in2) {
    const auto in1_mds = core::to_const_mdspan(std::forward<in1_t>(in1));
    const auto in2_mds = core::to_const_mdspan(std::forward<in2_t>(in2));
    using in1_mds_t = decltype(in1_mds);
    using in2_mds_t = decltype(in2_mds);

    if constexpr (in1_mds_t::rank() != in2_mds_t::rank()) {
        return false;
    }

    for (size_t i = 0; i < in1_mds_t::rank(); i++) {
        if (in1_mds.extent(i) != in2_mds.extent(i)) {
            return false;
        }
    }

    if constexpr (in1_mds_t::rank() == 0) {
        return in1_mds() == in2_mds();

    } else {
        for (typename in1_mds_t::index_type i = 0; i < in1_mds.extent(0); i++) {
            if (!array_equal(core::submdspan_from_left(in1_mds, i),
                             core::submdspan_from_left(in2_mds, i))) {
                return false;
            }
        }
    }

    return true;
}

} // namespace mdtensor
