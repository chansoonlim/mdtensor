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
namespace ufunc {

template <bool equal_nan>
constexpr bool array_equal_ufunc(auto &&in1, auto &&in2) {
    if constexpr (equal_nan && requires {
                      { std::isnan(in1) } -> std::convertible_to<bool>;
                      { std::isnan(in2) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in1) && std::isnan(in2)) {
            return true;
        }
    }

    using calc_t = core::common_data_type_t<decltype(in1), decltype(in2)>;

    return static_cast<calc_t>(in1) == static_cast<calc_t>(in2);
}

} // namespace ufunc

namespace {

template <bool equal_nan>
[[nodiscard]] constexpr bool array_equal_impl_(auto &&in1, auto &&in2) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    if constexpr (in1_mds.rank() == 0) {
        return ufunc::array_equal_ufunc<equal_nan>(in1_mds(), in2_mds());

    } else {
        using index_t = typename decltype(in1_mds)::index_type;

        for (index_t i = 0; i < in1_mds.extent(0); ++i) {
            if (!array_equal_impl_<equal_nan>(
                    core::submdspan_from_left(in1_mds, i),
                    core::submdspan_from_left(in2_mds, i))) {
                return false;
            }
        }

        return true;
    }
}

} // namespace

[[nodiscard]] constexpr bool array_equal(auto &&in1, auto &&in2,
                                         const bool equal_nan = false) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    if constexpr (in1_mds.rank() != in2_mds.rank()) {
        return false;

    } else {
        if (!core::is_same_extents(in1_mds.extents(), in2_mds.extents())) {
            return false;
        }

        if (equal_nan) {
            return array_equal_impl_<true>(in1_mds, in2_mds);

        } else {
            return array_equal_impl_<false>(in1_mds, in2_mds);
        }
    }
}

} // namespace mdtensor
