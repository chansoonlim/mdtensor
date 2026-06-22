/**
 * @file
 * @brief Element-wise clipping utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace detail {

template <typename in_t, typename min_t, typename max_t, typename out_t>
inline constexpr void clip_impl(in_t &&in, min_t &&min, max_t &&max,
                                out_t &&out) {
    // NOTE: std::clamp is not used to match the behavior with original np.clip
    out() = in();

    if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(min())>,
                                  std::nullopt_t>) {
        using value_t =
            core::data_common_type_t<std::remove_cvref_t<decltype(out())>,
                                     std::remove_cvref_t<decltype(min())>>;

        out() =
            std::max(static_cast<value_t>(out()), static_cast<value_t>(min()));
    }

    if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(max())>,
                                  std::nullopt_t>) {
        using value_t =
            core::data_common_type_t<std::remove_cvref_t<decltype(out())>,
                                     std::remove_cvref_t<decltype(max())>>;

        out() =
            std::min(static_cast<value_t>(out()), static_cast<value_t>(max()));
    }
}

} // namespace detail

/**
 * @brief Clip input values to the given range element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param min Minimum bound (mdspan, mdarray, scalar, etc.) or a disabled bound.
 * @param max Maximum bound (mdspan, mdarray, scalar, etc.) or a disabled bound.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = clip(in, min, max) in terms of array broadcasting.
 * @note Either bound may be disabled by passing a "no-bound" value (e.g.,
 *       nullopt wrapper), in which case only the other bound is applied.
 *
 * @see mdtensor::clip for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename min_t,
          typename max_t, typename out_t>
inline constexpr void clip_to(in_t &&in, min_t &&min, max_t &&max,
                              out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::clip_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_const_mdspan(std::forward<min_t>(min)),
        core::to_const_mdspan(std::forward<max_t>(max)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Clip input values to the given range element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         inputs.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param min Minimum bound (mdspan, mdarray, scalar, etc.) or a disabled bound.
 * @param max Maximum bound (mdspan, mdarray, scalar, etc.) or a disabled bound.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = clip(in, min, max) in terms of array broadcasting.
 * @note Either bound may be disabled by passing a "no-bound" value (e.g.,
 *       nullopt wrapper), in which case only the other bound is applied.
 *
 * @see mdtensor::clip_to for the in-place version that writes into an output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t,
          typename min_t, typename max_t>
[[nodiscard]] inline constexpr auto clip(in_t &&in, min_t &&min, max_t &&max) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::clip_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_const_mdspan(std::forward<min_t>(min)),
        core::to_const_mdspan(std::forward<max_t>(max)));
}

} // namespace mdtensor
