/**
 * @file
 * @brief Radian-to-degree conversion utilities for mdtensor.
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
[[nodiscard]] constexpr auto rad2deg(auto &&in,
                                     out_t &&out = out_t{std::nullopt},
                                     where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = core::resolve_output_like<dtype, true>(
        std::forward<decltype(out)>(out), in_mds);

    using calc_t =
        core::common_value_type_t<decltype(in_mds), decltype(out_md)>;

    static_assert(std::is_floating_point_v<calc_t> &&
                  "rad2deg conversion requires at least float precision.");

    constexpr calc_t R2D = std::numbers::inv_pi_v<calc_t> * calc_t{180};

    static_cast<void>(multiply<void, backend>(
        in_mds, R2D, out_md, std::forward<decltype(where)>(where)));

    return out_md;
}

} // namespace mdtensor
