/**
 * @file
 * @brief Eigen extent interop utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <Eigen/Dense>

#include "../../base/base.hpp"

namespace mdtensor::core::eigen {

template <std::size_t Extent>
[[nodiscard]] consteval int to_eigen_static_extent() {
    static_assert(
        Extent == core::dyn ||
            Extent <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "Static extent value exceeds maximum int value for Eigen mapping.");

    if constexpr (Extent == core::dyn) {
        return Eigen::Dynamic;

    } else {
        return static_cast<int>(Extent);
    }
}

template <core::integral_c ext_t>
[[nodiscard]] constexpr Eigen::Index to_eigen_extent(ext_t &&ext) {
    if constexpr (std::signed_integral<ext_t>) {
        if (ext < 0) {
            throw std::invalid_argument(
                "Negative extent value is invalid for Eigen mapping.");
        }
    }

    if (!std::in_range<Eigen::Index>(ext)) {
        throw std::invalid_argument(
            "Extent value exceeds maximum int value for Eigen mapping.");
    }

    return static_cast<Eigen::Index>(ext);
}

} // namespace mdtensor::core::eigen
