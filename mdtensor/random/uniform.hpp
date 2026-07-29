/**
 * @file
 * @brief Uniform random distribution utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../math/add.hpp"
#include "../math/multiply.hpp"
#include "rand.hpp"

namespace mdtensor {
namespace random {

template <core::MPMode mpmode = core::MPMode::NONE, typename in_t>
inline constexpr void uniform_to(in_t &&in, const double &low = 0,
                                 const double &high = 1) noexcept {
    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));

    using T = typename decltype(in_mds)::value_type;

    random::rand_to(in_mds);

    if constexpr (mpmode == core::MPMode::SIMD) {
        multiply_to<mpmode>(in_mds, static_cast<const T>(high - low), in_mds);
        add_to<mpmode>(in_mds, static_cast<const T>(low), in_mds);
        return;

    } else {
        core::batch<mpmode>(
            [&](auto &&in) {
                in() = static_cast<const T>(high - low) * in() +
                       static_cast<const T>(low);
            },
            std::integer_sequence<bool, true>{}, in_mds);
    }
}

template <std::floating_point dtype = float,
          core::MPMode mpmode = core::MPMode::NONE,
          core::extents_c exts_t = core::extents<uint8_t>>
[[nodiscard]] inline constexpr auto uniform(exts_t &&exts = exts_t{},
                                            const double &low = 0,
                                            const double &high = 1) noexcept {
    auto out = empty<dtype>(std::forward<exts_t>(exts));
    uniform_to<mpmode>(out, low, high);
    return out;
}

} // namespace random
} // namespace mdtensor
