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

/**
 * @brief Fill an output tensor with uniform random values in [low, high)
 *        (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Output mdspan, mdarray, scalar, etc.
 * @param low Lower bound of the uniform distribution (inclusive).
 * @param high Upper bound of the uniform distribution (exclusive).
 *
 * @note The underlying generator produces values in [0, 1), which are then
 *       scaled and shifted to the target range.
 * @note If high == low, the output becomes constant equal to low.
 * @note Behavior for high < low follows the affine transform and is not
 *       explicitly validated.
 *
 * @see mdtensor::random::uniform for the out-of-place version that allocates an
 *      output and fills it.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t>
inline constexpr void uniform_to(in_t &&in, const double &low = 0,
                                 const double &high = 1) noexcept {
    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));

    using T = typename decltype(in_mds)::value_type;

    random::rand_to(in_mds);

    if constexpr (mpmode == MPMode::SIMD) {
        multiply_to(in_mds, static_cast<const T>(high - low), in_mds);
        add_to(in_mds, static_cast<const T>(low), in_mds);
        return;

    } else {
        core::batch<mpmode>(
            [&](auto &&in) {
                in() = static_cast<const T>(high - low) * in() +
                       static_cast<const T>(low);
            },
            in_mds);
    }
}

/**
 * @brief Create an array of uniform random values in [low, high)
 *        (out-of-place).
 *
 * @tparam dtype (optional) Floating-point element type. Default is float.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam exts_t (optional) Extents type. Default is extents<uint8_t>.
 *
 * @param exts Output extents.
 * @param low Lower bound of the uniform distribution (inclusive).
 * @param high Upper bound of the uniform distribution (exclusive).
 *
 * @return Newly allocated mdarray filled with uniform random values in [low,
 *         high).
 *
 * @see mdtensor::random::uniform_to for the in-place version that fills an
 *      existing output.
 */
template <std::floating_point dtype = float, MPMode mpmode = MPMode::NONE,
          extents_c exts_t = extents<uint8_t>>
[[nodiscard]] inline constexpr auto uniform(exts_t &&exts = exts_t{},
                                            const double &low = 0,
                                            const double &high = 1) noexcept {
    auto out = empty<dtype>(std::forward<exts_t>(exts));
    uniform_to<mpmode>(out, low, high);
    return out;
}

} // namespace random
} // namespace mdtensor
