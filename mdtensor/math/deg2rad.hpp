/**
 * @file
 * @brief Degree-to-radian conversion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "multiply.hpp"

namespace mdtensor {

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto deg2rad(auto &&in,
                                     out_t &&out = out_t{std::nullopt},
                                     where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using calc_t = core::floating_calc_type_t<dtype, decltype(in_mds)>;

    auto out_md = core::resolve_output_like<calc_t, true>(
        std::forward<decltype(out)>(out), in_mds);

    constexpr calc_t D2R = std::numbers::pi_v<calc_t> / calc_t{180};

    static_cast<void>(multiply<void, backend>(
        in_mds, D2R, out_md, std::forward<decltype(where)>(where)));

    return out_md;
}

} // namespace mdtensor
