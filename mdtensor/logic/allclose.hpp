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
#if false
    // NOTE: This implementation is not used because it cannot use escape.
    // TODO: use lazy-evaluation
    return all<false, core::Backend::NATIVE>(isclose<backend>(
        std::forward<decltype(in1)>(in1), std::forward<decltype(in2)>(in2),
        std::forward<decltype(rtol)>(rtol),
        std::forward<decltype(atol)>(atol), std::nullopt, equal_nan));

#else
    bool init = false;

    const bool is_allclose = core::batch_while(
        [&](auto &&in1_u, auto &&in2_u, auto &&rtol_u, auto &&atol_u) {
            static_cast<void>(core::initialize_ufunc(init, std::nullopt));

            bool out;
            ufunc::isclose_ufunc(std::forward<decltype(in1_u)>(in1_u),
                                 std::forward<decltype(in2_u)>(in2_u), out,
                                 std::forward<decltype(rtol_u)>(rtol_u),
                                 std::forward<decltype(atol_u)>(atol_u),
                                 equal_nan);
            return out;
        },
        std::integer_sequence<bool, true, true, true, true>{},
        std::forward<decltype(in1)>(in1), std::forward<decltype(in2)>(in2),
        std::forward<decltype(rtol)>(rtol), std::forward<decltype(atol)>(atol));

    if (!init) {
        throw std::runtime_error(
            "mdtensor::allclose: cannot initialize output tensor.");
    }

    return is_allclose;

#endif
}

} // namespace mdtensor
