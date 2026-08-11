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

constexpr void uniform_ufunc(auto &&out, auto &&low, auto &&high,
                             auto &&engine) {
    using value_t = std::remove_cvref_t<decltype(out())>;

    rand_ufunc(std::forward<decltype(out)>(out),
               std::forward<decltype(engine)>(engine));

    out() =
        (static_cast<value_t>(high()) - static_cast<value_t>(low())) * out() +
        static_cast<value_t>(low());
}

} // namespace ufunc

template <
    typename dtype = double, typename EngineType = default_random_engine_t,
    typename shape_t = core::extents<std::uint8_t>, typename low_t = dtype,
    typename high_t = dtype, typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto
uniform(shape_t &&shape = shape_t{}, low_t &&low = low_t{0},
        high_t &&high = high_t{1}, out_t &&out = out_t{std::nullopt},
        const seed_t seed = make_random_seed()) {
    const auto low_mds =
        core::to_const_mdspan(std::forward<decltype(low)>(low));
    const auto high_mds =
        core::to_const_mdspan(std::forward<decltype(high)>(high));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::to_extents(std::forward<decltype(shape)>(shape)), low_mds,
                high_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    static_assert(core::floating_point_c<
                      typename core::to_mdspan_t<decltype(out_md)>::value_type>,
                  "Output must have a floating point value type.");

    auto engine = generator::EngineWrapper<EngineType>{seed.value};

    core::batch_with_broadcast<core::Backend::NATIVE>(
        [&](auto &&...elems) {
            ufunc::uniform_ufunc(std::forward<decltype(elems)>(elems)...,
                                 engine);
        },
        std::integer_sequence<bool, false, true, true>{}, out_md, low_mds,
        high_mds);

    return out_md;
}

} // namespace mdtensor::random
