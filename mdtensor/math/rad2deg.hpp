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

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            // NOTE: ensure that the output type is at least float precision
            using value_t = core::output_value_t<
                dtype, typename decltype(in_mds)::value_type, float>;

            return core::make_output<value_t>(core::extents<std::uint8_t>{},
                                              in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    using calc_t = core::common_data_type_t<
        typename decltype(in_mds)::value_type,
        typename core::to_mdspan_t<decltype(out_md)>::value_type>;

    static_assert(std::is_floating_point_v<calc_t> &&
                  "rad2deg conversion requires at least float precision.");

    constexpr calc_t R2D = std::numbers::inv_pi_v<calc_t> * calc_t{180};

    static_cast<void>(multiply<void, backend>(
        in_mds, R2D, out_md, std::forward<decltype(where)>(where)));

    return out_md;
}

} // namespace mdtensor
