/**
 * @file
 * @brief All-close comparison utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "all.hpp"
#include "isclose.hpp"

namespace mdtensor {

template <core::Backend backend = core::Backend::AUTO, typename rtol_t = double,
          typename atol_t = double>
[[nodiscard]] constexpr bool
allclose(auto &&in1, auto &&in2, rtol_t &&rtol = rtol_t{1e-05},
         atol_t &&atol = atol_t{1e-08}, const bool equal_nan = false) {
    return all<void, false, core::Backend::NATIVE>(isclose<bool, backend>(
        std::forward<decltype(in1)>(in1), std::forward<decltype(in2)>(in2),
        std::forward<decltype(rtol)>(rtol), std::forward<decltype(atol)>(atol),
        std::nullopt, equal_nan));
}

} // namespace mdtensor
