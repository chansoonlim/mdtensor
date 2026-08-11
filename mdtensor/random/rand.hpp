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

template <std::floating_point value_t>
[[nodiscard]] constexpr value_t pow2_neg(std::size_t bits) noexcept {
    value_t result = value_t{1};

    for (std::size_t i = 0; i < bits; i++) {
        result *= value_t{0.5};
    }

    return result;
}

constexpr void rand_ufunc(auto &&out, auto &&engine) {
    using value_t = std::remove_cvref_t<decltype(out)>;

    static_assert(std::is_floating_point_v<value_t>,
                  "rand_ufunc requires floating-point value type.");

    using base_t = typename std::remove_cvref_t<decltype(engine)>::base_t;

    static_assert(std::unsigned_integral<base_t>,
                  "rand_ufunc requires an unsigned integral engine result.");

    constexpr std::size_t base_bits = std::numeric_limits<base_t>::digits;
    constexpr std::size_t value_bits = std::numeric_limits<value_t>::digits;

    if constexpr (value_bits <= base_bits) {
        const base_t bits = engine() >> (base_bits - value_bits);
        out = static_cast<value_t>(bits) * pow2_neg<value_t>(value_bits);

    } else {
        value_t result = value_t{0};
        value_t scale = value_t{1};

        std::size_t remaining = value_bits;

        while (remaining > 0) {
            const std::size_t take =
                remaining < base_bits ? remaining : base_bits;

            const base_t bits = engine() >> (base_bits - take);

            scale *= pow2_neg<value_t>(take);
            result += static_cast<value_t>(bits) * scale;

            remaining -= take;
        }

        out = result;
    }
}

} // namespace ufunc

template <typename dtype = double,
          typename EngineType = default_random_engine_t,
          typename shape_t = core::extents<std::uint8_t>,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto rand(shape_t &&shape = shape_t{},
                                  out_t &&out = out_t{std::nullopt},
                                  const seed_t seed = make_random_seed()) {
    auto out_md = core::resolve_output<dtype, true>(
        std::forward<decltype(out)>(out),
        core::to_extents(std::forward<decltype(shape)>(shape)));

    auto engine = generator::EngineWrapper<EngineType>{seed.value};

    core::batch<core::Backend::NATIVE,
                core::to_mdspan_t<decltype(out_md)>::rank()>(
        [&](auto &&...elems) {
            ufunc::rand_ufunc(std::forward<decltype(elems)>(elems)..., engine);
        },
        out_md);

    return out_md;
}

} // namespace mdtensor::random
