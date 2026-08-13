/**
 * @file
 * @brief Uniform random distribution utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "rand.hpp"

namespace mdtensor::random {
namespace ufunc {

template <typename dtype = void>
constexpr void uniform_ufunc(auto &&out, auto &&low, auto &&high,
                             auto &&engine) {
    using calc_t = core::floating_calc_type_t<dtype, decltype(out),
                                              decltype(low), decltype(high)>;

    rand_ufunc<calc_t>(std::forward<decltype(out)>(out),
                       std::forward<decltype(engine)>(engine));

    out = static_cast<calc_t>(low) +
          (static_cast<calc_t>(high) - static_cast<calc_t>(low)) *
              static_cast<calc_t>(out);
}

} // namespace ufunc

template <std::floating_point value_t = double,
          typename EngineType = default_random_engine_t,
          typename shape_t = core::extents<std::uint8_t>,
          typename low_t = value_t, typename high_t = value_t,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto
uniform(shape_t &&shape = shape_t{}, low_t &&low = low_t{0},
        high_t &&high = high_t{1}, out_t &&out = out_t{std::nullopt},
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
            ufunc::uniform_ufunc<value_t>(
                std::forward<decltype(elems)>(elems)..., engine);
        },
        std::integer_sequence<bool, false, true, true>{}, out_md, low_mds,
        high_mds);

    return out_md;
}

} // namespace mdtensor::random
