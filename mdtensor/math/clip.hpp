/**
 * @file
 * @brief Element-wise clipping utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../creation/empty_like.hpp"

namespace mdtensor {
namespace ufunc {

constexpr void clip_ufunc(auto &&in, auto &&min, auto &&max, auto &&out) {
    // NOTE: std::clamp is not used to match the behavior with original np.clip
    // when min > max, np.clip returns max, and std::clamp returns min.
    // mdtensor.clip is designed to match the behavior of np.clip.

    using value_t = std::remove_cvref_t<decltype(in())>;

    out() = in();

    if constexpr (!core::is_nullopt_t_c<decltype(min())>) {
        out() = std::max(out(), static_cast<value_t>(min()));
    }

    if constexpr (!core::is_nullopt_t_c<decltype(max())>) {
        out() = std::min(out(), static_cast<value_t>(max()));
    }
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename min_t = std::nullopt_t, typename max_t = std::nullopt_t,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto clip(auto &&in, min_t &&min = min_t{std::nullopt},
                                  max_t &&max = max_t{std::nullopt},
                                  out_t &&out = out_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::is_nullopt_t_c<decltype(out)>) {
            return empty_like<dtype>(in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::clip_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, true, false>{}, in_mds,
        std::forward<decltype(min)>(min), std::forward<decltype(max)>(max),
        out_md);

    return out_md;
}

} // namespace mdtensor
