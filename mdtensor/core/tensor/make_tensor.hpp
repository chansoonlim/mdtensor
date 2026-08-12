/**
 * @file
 * @brief Make tensor utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "type.hpp"

namespace mdtensor::core {

template <typename value_t = double, extents_c exts_t>
[[nodiscard]] constexpr auto make_tensor(exts_t &&exts) {
    using base_t = std::remove_cvref_t<decltype(exts)>;

    if constexpr (base_t::rank() == 0) {
        return tensor<value_t, base_t>{};

    } else {
        return tensor<value_t, base_t>{std::forward<exts_t>(exts)};
    }
}

template <typename dtype = void>
[[nodiscard]] constexpr auto make_tensor_like(auto &&in) {
    const auto in_mds = to_const_mdspan(std::forward<decltype(in)>(in));

    using value_t = output_value_t<dtype, decltype(in_mds)>;

    return make_tensor<value_t>(in_mds.extents());
}

template <typename T>
using make_tensor_like_t = decltype(make_tensor_like(std::declval<T>()));

template <typename dtype = void, bool floating = false, extents_c exts_t>
[[nodiscard]] constexpr auto resolve_output(auto &&out, exts_t &&exts) {
    if constexpr (floating) {
        // Ensure that the output type is at least float precision
        using value_t = core::output_value_t<dtype, float>;

        const auto out_md = resolve_output<value_t, false>(
            std::forward<decltype(out)>(out), std::forward<exts_t>(exts));

        // Check that resolved output type is at least float precision
        static_assert(core::floating_point_c<
                          typename to_mdspan_t<decltype(out_md)>::value_type>,
                      "Resolved output type must be at least float precision.");

        return out_md;

    } else if constexpr (core::nullopt_t_c<decltype(out)>) {
        return make_tensor<dtype>(std::forward<exts_t>(exts));

    } else {
        const auto out_mds = to_output_mdspan(std::forward<decltype(out)>(out));

        using value_t = typename decltype(out_mds)::value_type;
        using calc_t = core::output_value_t<dtype, value_t>;

        static_assert(
            std::same_as<core::common_value_type_t<calc_t, value_t>, value_t>,
            "Resolved output type must not be less precise than desired.");

        static_assert(
            !is_always_different_extents<decltype(out_mds.extents()), exts_t>(),
            "Output tensor extents must match the provided extents.");

        if (!is_same_extents(out_mds.extents(), std::forward<exts_t>(exts))) {
            throw std::invalid_argument("Provided output tensor extents do "
                                        "not match the expected extents.");
        }

        return out_mds;
    }
}

template <typename dtype = void, bool floating = false>
[[nodiscard]] constexpr auto resolve_output_like(auto &&out, auto &&in) {
    const auto in_mds = to_const_mdspan(std::forward<decltype(in)>(in));

    if constexpr (floating) {
        // Ensure that the output type is at least float precision
        using value_t = core::output_value_t<dtype, decltype(in_mds), float>;

        return resolve_output<value_t, floating>(
            std::forward<decltype(out)>(out), in_mds.extents());

    } else {
        using value_t = core::output_value_t<dtype, decltype(in_mds)>;

        return resolve_output<value_t, floating>(
            std::forward<decltype(out)>(out), in_mds.extents());
    }
}

} // namespace mdtensor::core
