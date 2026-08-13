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

    out = in;

    if constexpr (!core::nullopt_t_c<decltype(min)>) {
        using common_t =
            core::common_arithmetic_type_t<decltype(out), decltype(min)>;

        out = std::max(static_cast<common_t>(out), static_cast<common_t>(min));
    }

    if constexpr (!core::nullopt_t_c<decltype(max)>) {
        using common_t =
            core::common_arithmetic_type_t<decltype(out), decltype(max)>;

        out = std::min(static_cast<common_t>(out), static_cast<common_t>(max));
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
    const auto min_mds =
        core::to_const_mdspan(std::forward<decltype(min)>(min));
    const auto max_mds =
        core::to_const_mdspan(std::forward<decltype(max)>(max));

    using calc_t =
        core::floating_calc_type_t<dtype, decltype(in_mds), decltype(min_mds),
                                   decltype(max_mds)>;

    auto out_md = core::resolve_broadcasted_output<calc_t>(
        std::forward<decltype(out)>(out), core::extents<std::uint8_t>{}, in_mds,
        min_mds, max_mds);

    core::batch<backend>(
        [](auto &&...elems) {
            ufunc::clip_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, true, false>{}, in_mds,
        std::forward<decltype(min)>(min), std::forward<decltype(max)>(max),
        out_md);

    return out_md;
}

} // namespace mdtensor
