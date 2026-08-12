/**
 * @file
 * @brief Range generation utilities for mdtensor (arange).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "empty.hpp"

namespace mdtensor {

template <typename dtype = void, core::extents_c exts_t,
          core::arithmetic_c start_t = int, core::arithmetic_c step_t = int,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto arange(exts_t &&exts, start_t &&start = start_t{0},
                                    step_t &&step = step_t{1},
                                    out_t &&out = out_t{std::nullopt}) {
    static_assert(exts.rank() == 1, "arange only supports rank-1 extents");

    using calc_t = core::output_value_t<dtype, start_t, step_t>;

    auto out_md = core::resolve_output<calc_t>(
        std::forward<decltype(out)>(out), std::forward<decltype(exts)>(exts));

    using value_t = decltype(out_md)::value_type;
    using index_t = decltype(out_md)::index_type;

    const calc_t actual_step =
        static_cast<calc_t>(start + step) - static_cast<calc_t>(start);

    out_md(0) = static_cast<value_t>(start);

    for (index_t i = 1; i < out_md.extent(0); i++) {
        out_md(i) = out_md(i - 1) + static_cast<value_t>(actual_step);
    }

    return out_md;
}

template <typename dtype = void, core::arithmetic_c start_t,
          core::arithmetic_c stop_t, core::arithmetic_c step_t = int,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto arange(start_t &&start, stop_t &&stop,
                                    step_t &&step = step_t{1},
                                    out_t &&out = out_t{std::nullopt}) {
    const std::int64_t num = std::ceil((stop - start) / step);

    if (num < 0) {
        throw std::invalid_argument(
            "calculated number of elements is negative");
    }

    return arange<dtype>(core::dims<1>{static_cast<std::size_t>(num)},
                         std::forward<start_t>(start),
                         std::forward<step_t>(step), std::forward<out_t>(out));
}

template <typename dtype = void, core::arithmetic_c stop_t>
[[nodiscard]] constexpr auto arange(stop_t &&stop) {
    using start_t = typename std::remove_cvref_t<stop_t>;

    return arange<dtype>(start_t{0}, std::forward<stop_t>(stop), start_t{1});
}

template <std::size_t num, typename dtype = void,
          core::arithmetic_c start_t = int, core::arithmetic_c step_t = int,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto arange(start_t &&start = start_t{0},
                                    step_t &&step = step_t{1},
                                    out_t &&out = out_t{std::nullopt}) {
    return arange<dtype>(core::extents<std::size_t, num>{},
                         std::forward<start_t>(start),
                         std::forward<step_t>(step), std::forward<out_t>(out));
}

} // namespace mdtensor
