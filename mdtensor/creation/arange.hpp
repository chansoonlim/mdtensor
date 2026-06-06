/**
 * @file
 * @brief Range generation utilities for mdtensor (arange).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {

/**
 * @brief Generate a 1-D range of values with a given step (out-of-place).
 *
 * @tparam data_t (optional) Output value type. If void, deduced from inputs.
 * @tparam start_t Arithmetic type of start.
 * @tparam stop_t Arithmetic type of stop.
 * @tparam step_t (optional) Arithmetic type of step. Default is double.
 *
 * @param start Start value (inclusive).
 * @param stop Stop value (exclusive in intent; number of elements is computed
 *             from (stop - start) / step).
 * @param step Step size. Default is 1.
 *
 * @return 1-D mdarray containing the generated range.
 *
 * @note The output length is computed as ceil((stop - start) / step).
 * @note The effective step is computed in the output value type to reduce
 *       accumulation drift when `start_t`/`step_t` differ from `data_t`.
 *
 * @warning This implementation does not currently guard against invalid input
 *          such as step == 0 or mismatched sign between (stop - start) and
 *          step. Callers should ensure the range definition is valid.
 */
template <typename data_t = void, arithmetic_c start_t, arithmetic_c stop_t,
          arithmetic_c step_t = double>
[[nodiscard]] inline constexpr auto arange(start_t &&start, stop_t &&stop,
                                           step_t &&step = (step_t)1) noexcept {
    using value_t =
        std::conditional_t<!std::is_void_v<data_t>, data_t,
                           core::data_common_type_t<start_t, stop_t>>;

    const size_t num = std::ceil((stop - start) / step);
    const value_t step_actual =
        static_cast<value_t>(start + step) - static_cast<value_t>(start);

    auto out = mdarray<value_t, dims<1>>{dims<1>{num}};

    out(0) = start;
    for (size_t i = 1; i < num; i++) {
        out(i) = out(i - 1) + step_actual;
    }

    return out;
}

/**
 * @brief Generate a 1-D range [0, stop) with step 1 (out-of-place).
 *
 * @tparam data_t (optional) Output value type. If void, deduced from input.
 * @tparam stop_t Arithmetic type of stop.
 *
 * @param stop Stop value (exclusive in intent).
 *
 * @return 1-D mdarray containing the generated range.
 *
 * @see mdtensor::arange(start, stop, step) for the general form.
 */
template <typename data_t = void, arithmetic_c stop_t>
[[nodiscard]] inline constexpr auto arange(stop_t &&stop) noexcept {
    return arange<data_t>(0, std::forward<stop_t>(stop));
}

} // namespace mdtensor
