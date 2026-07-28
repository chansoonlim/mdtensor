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
            core::common_data_type_t<std::remove_cvref_t<decltype(out())>,
                                     std::remove_cvref_t<decltype(min())>>;

        out() =
            std::max(static_cast<value_t>(out()), static_cast<value_t>(min()));
    }

    if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(max())>,
                                  std::nullopt_t>) {
        using value_t =
            core::common_data_type_t<std::remove_cvref_t<decltype(out())>,
                                     std::remove_cvref_t<decltype(max())>>;

        out() =
            std::min(static_cast<value_t>(out()), static_cast<value_t>(max()));
    }
}

} // namespace detail

template <MPMode mpmode = MPMode::NONE, typename in_t, typename min_t,
          typename max_t, typename out_t>
inline constexpr void clip_to(in_t &&in, min_t &&min, max_t &&max,
                              out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::clip_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, false, false, false, true>{},
        std::forward<in_t>(in), std::forward<min_t>(min),
        std::forward<max_t>(max), std::forward<out_t>(out));
}

template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t,
          typename min_t, typename max_t>
[[nodiscard]] inline constexpr auto clip(in_t &&in, min_t &&min, max_t &&max) {
    auto out = core::create_out<dtype>(
        extents<uint8_t>{}, std::forward<in_t>(in), std::forward<min_t>(min),
        std::forward<max_t>(max));

    clip_to<mpmode>(std::forward<in_t>(in), std::forward<min_t>(min),
                    std::forward<max_t>(max), out);

    return out;
}

} // namespace mdtensor
