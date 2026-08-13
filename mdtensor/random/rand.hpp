/**
 * @file
 * @brief Random number generation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../creation/empty.hpp"
#include "randint.hpp"

namespace mdtensor::random {
namespace ufunc {
namespace detail {

template <std::floating_point calc_t>
[[nodiscard]] constexpr calc_t pow2_neg(std::size_t bits) noexcept {
    calc_t result = calc_t{1};

    for (std::size_t i = 0; i < bits; i++) {
        result *= calc_t{0.5};
    }

    return result;
}

} // namespace detail

template <typename dtype = void>
constexpr void rand_ufunc(auto &&out, auto &&engine) {
    using base_t = typename std::remove_cvref_t<decltype(engine)>::base_t;

    static_assert(std::unsigned_integral<base_t>,
                  "rand_ufunc requires an unsigned integral engine result.");

    using calc_t = core::floating_calc_type_t<dtype, decltype(out)>;

    constexpr std::size_t base_bits = std::numeric_limits<base_t>::digits;
    constexpr std::size_t value_bits = std::numeric_limits<calc_t>::digits;

    if constexpr (value_bits <= base_bits) {
        const base_t bits = engine() >> (base_bits - value_bits);
        out = static_cast<calc_t>(bits) * detail::pow2_neg<calc_t>(value_bits);

    } else {
        calc_t result = 0;
        calc_t scale = 1;

        std::size_t remaining = value_bits;

        while (remaining > 0) {
            const std::size_t take =
                remaining < base_bits ? remaining : base_bits;

            const base_t base = engine() >> (base_bits - take);

            scale *= detail::pow2_neg<calc_t>(static_cast<calc_t>(take));
            result += static_cast<calc_t>(base) * scale;

            remaining -= take;
        }

        out = result;
    }
}

} // namespace ufunc

template <std::floating_point value_t = double,
          typename EngineType = default_random_engine_t,
          typename shape_t = core::extents<std::uint8_t>,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto rand(shape_t &&shape = shape_t{},
                                  out_t &&out = out_t{std::nullopt},
                                  const seed_t seed = make_random_seed()) {
    auto out_md = core::resolve_output<value_t, true>(
        std::forward<decltype(out)>(out),
        core::to_extents(std::forward<decltype(shape)>(shape)));

    auto engine = generator::EngineWrapper<EngineType>{seed.value};

    core::batch<core::Backend::NATIVE>(
        [&](auto &&...elems) {
            ufunc::rand_ufunc<value_t>(std::forward<decltype(elems)>(elems)...,
                                       engine);
        },
        out_md);

    return out_md;
}

} // namespace mdtensor::random
