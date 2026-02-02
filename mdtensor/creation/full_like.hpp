/**
 * @file
 * @brief Full-like creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "empty_like.hpp"
#include "fill.hpp"

namespace mdtensor {

/**
 * @brief Create a new tensor filled with a scalar value, matching an input's
 *        shape (out-of-place).
 *
 * @tparam dtype (optional) Data type of the result. If void, deduced from the
 * input value type.
 * @tparam mpmode (optional) Parallel execution mode used for filling. Default
 * is MPMode::NONE.
 *
 * @param in Reference tensor whose extents are used (mdspan, mdarray, etc.).
 * @param val Fill value.
 *
 * @return Newly allocated tensor (mdarray) with the same extents as `in`,
 * filled with `val`.
 *
 * @note Equivalent to `out = empty_like(in); out[...] = val`.
 *
 * @see mdtensor::empty_like
 * @see mdtensor::fill
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t,
          typename val_t>
[[nodiscard]] inline constexpr auto full_like(in_t &&in, val_t &&val) {
    auto out = empty_like<dtype>(std::forward<in_t>(in));
    fill<mpmode>(out, std::forward<val_t>(val));
    return out;
}

} // namespace mdtensor
