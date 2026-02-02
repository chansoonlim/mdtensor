/**
 * @file
 * @brief Copy-into-like utilities for mdtensor (copy_like).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "copy.hpp"
#include "empty_like.hpp"

namespace mdtensor {

/**
 * @brief Allocate an output like `in1` and copy `in2` into it (out-of-place).
 *
 * @tparam dtype (optional) Output value type. If void, deduced by
 *         empty_like.
 * @tparam mpmode (optional) Parallel execution mode for the copy operation.
 *         Default is MPMode::NONE.
 *
 * @param in1 Shape/layout reference (mdspan, mdarray, etc.). The output is
 *            allocated to be "like" this input.
 * @param in2 Source data to copy from (mdspan, mdarray, scalar, etc.).
 *
 * @return Newly allocated container (typically mdarray-like) that has the same
 *         shape/layout as `in1` and contains the copied values from `in2`.
 *
 * @note This is equivalent to:
 *       - out = empty_like(in1)
 *       - copy_to(in2, out)
 *       in terms of allocation + broadcasting semantics provided by copy_to.
 *
 * @see mdtensor::empty_like
 * @see mdtensor::copy_to
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto copy_like(in1_t &&in1, in2_t &&in2) {
    auto out = empty_like<dtype>(std::forward<in1_t>(in1));
    copy_to<mpmode>(std::forward<in2_t>(in2), out);
    return out;
}

} // namespace mdtensor
