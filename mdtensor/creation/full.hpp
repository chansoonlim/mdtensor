/**
 * @file
 * @brief Full creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "empty.hpp"
#include "fill.hpp"

namespace mdtensor {

/**
 * @brief Create a new tensor filled with a scalar value (out-of-place).
 *
 * @tparam dtype Element type of the result tensor.
 * @tparam exts_t (optional) Extents type. Default is extents<size_t>.
 * @tparam mpmode (optional) Parallel execution mode used for filling. Default
 * is MPMode::NONE.
 *
 * @param val Fill value.
 * @param exts Output extents.
 *
 * @return Newly allocated tensor (mdarray) with extents `exts`, filled with
 * `val`.
 *
 * @note Equivalent to `out = empty<dtype>(exts); fill(out, val);`.
 *
 * @see mdtensor::empty
 * @see mdtensor::fill
 */
template <typename dtype, extents_c exts_t = extents<size_t>,
          MPMode mpmode = MPMode::NONE>
[[nodiscard]] inline constexpr auto full(dtype &&val,
                                         exts_t &&exts = exts_t{}) {
    auto out = empty<dtype>(std::forward<exts_t>(exts));
    fill<mpmode>(out, std::forward<dtype>(val));
    return out;
}

} // namespace mdtensor
