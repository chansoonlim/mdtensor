/**
 * @file
 * @brief Random integer generation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <random>

#include "../creation/empty.hpp"
#include "generator/generator.hpp"
#include "seed.hpp"

namespace mdtensor::random {

using default_random_engine_t = std::mt19937_64;

namespace ufunc {

template <typename dtype = void>
constexpr void randint_ufunc(auto &&out, auto &&low, auto &&high,
                             auto &&engine) {
    using calc_t =
        core::calc_type_t<dtype, decltype(out), decltype(low), decltype(high)>;

    static_assert(std::is_integral_v<calc_t>,
                  "randint_ufunc requires integral calculation type.");

    constexpr bool has_low =
        !core::nullopt_t_c<std::remove_cvref_t<decltype(low)>>;
    constexpr bool has_high =
        !core::nullopt_t_c<std::remove_cvref_t<decltype(high)>>;

    if constexpr (has_low && has_high) {
        // NOTE: This implementation matches the behavior of
        // numpy.random.randint(low, high)

        out = engine.template get_bounded<calc_t>(static_cast<calc_t>(low),
                                                  static_cast<calc_t>(high));

    } else if constexpr (has_low && !has_high) {
        // NOTE: This implementation matches the behavior of
        // numpy.random.randint(low, high=None)

        out = engine.template get_bounded<calc_t>(calc_t{0},
                                                  static_cast<calc_t>(low));

    } else if constexpr (!has_low && has_high) {
        // NOTE: This implementation is not exist in numpy.random.randint,
        // but maybe useful for some use cases.

        out = engine.template get_bounded<calc_t>(
            std::numeric_limits<calc_t>::lowest(), static_cast<calc_t>(high));

    } else {
        // NOTE: This implementation is not exist in numpy.random.randint,
        // but maybe useful for some use cases.

        out = engine.template get<calc_t>();
    }
}

} // namespace ufunc

template <std::integral value_t = int,
          typename EngineType = default_random_engine_t,
          typename shape_t = core::extents<std::uint8_t>,
          typename low_t = std::nullopt_t, typename high_t = std::nullopt_t,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto
randint(shape_t &&shape = shape_t{}, low_t &&low = low_t{std::nullopt},
        high_t &&high = high_t{std::nullopt}, out_t &&out = out_t{std::nullopt},
        const seed_t seed = make_random_seed()) {
    const auto low_mds =
        core::to_const_mdspan(std::forward<decltype(low)>(low));
    const auto high_mds =
        core::to_const_mdspan(std::forward<decltype(high)>(high));

    auto out_md = core::resolve_output<value_t>(
        std::forward<decltype(out)>(out),
        core::get_broadcast_extents(
            std::index_sequence<0, 0, 0>{},
            core::to_extents(std::forward<decltype(shape)>(shape)),
            low_mds.extents(), high_mds.extents()));

    auto engine = generator::EngineWrapper<EngineType>{seed.value};

    core::batch<core::Backend::NATIVE>(
        [&](auto &&...elems) {
            ufunc::randint_ufunc<value_t>(
                std::forward<decltype(elems)>(elems)..., engine);
        },
        std::integer_sequence<bool, false, true, true>{}, out_md, low_mds,
        high_mds);

    return out_md;
}

} // namespace mdtensor::random
