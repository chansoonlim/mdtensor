/**
 * @file
 * @brief Random number generation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <cstdint>
#include <random>

#include "../creation/copy.hpp"
#include "../creation/empty.hpp"

namespace mdtensor {
namespace random {
namespace detail {

// Code adapted from:
// https://mklimenko.github.io/english/2018/06/04/constexpr-random/

constexpr std::uint32_t lce_a = 16807;
constexpr std::uint32_t lce_c = 0;
constexpr std::uint32_t lce_m = 2147483647;

[[nodiscard]] inline constexpr std::uint32_t time_from_string(const char *str,
                                                              int offset) {
    return static_cast<std::uint32_t>(str[offset] - '0') * 10 +
           static_cast<std::uint32_t>(str[offset + 1] - '0');
}

[[nodiscard]] inline constexpr std::uint32_t get_seed_constexpr() {
    const char *t = __TIME__;
    return time_from_string(t, 0) * 3600 + time_from_string(t, 3) * 60 +
           time_from_string(t, 6);
}

struct LCEngine {
    std::uint32_t state;

    constexpr LCEngine(std::uint32_t seed) : state(seed) {}

    [[nodiscard]] constexpr std::uint32_t next() {
        state = (lce_a * state + lce_c) % lce_m;
        return state;
    }

    [[nodiscard]] constexpr double next_normalized() {
        return static_cast<double>(next()) / lce_m;
    }
};

template <typename T, size_t sz>
[[nodiscard]] inline constexpr auto uniform_distribution(T min, T max) {
    std::array<T, sz> dst{};
    LCEngine rng{get_seed_constexpr()};
    for (auto &el : dst)
        el = static_cast<T>(rng.next_normalized() * (max - min) + min);
    return dst;
}

template <core::md_c in_t>
    requires(
        std::remove_cvref_t<in_t>::rank() == 0 &&
        std::floating_point<typename std::remove_cvref_t<in_t>::value_type>)
inline void rand_impl(in_t &&in) noexcept {
    using dist_t = std::uniform_real_distribution<
        typename std::remove_cvref_t<in_t>::value_type>;

    static auto rd = std::random_device{};
    thread_local static auto gen = std::mt19937(rd());
    static auto dist = dist_t{0, 1};

    in() = dist(gen);
}

} // namespace detail

template <core::MPMode mpmode = core::MPMode::NONE, typename in_t>
inline constexpr void rand_to(in_t &&in) noexcept {
    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    if constexpr (in_mds_t::rank_dynamic() == 0) {
        if (std::is_constant_evaluated()) {
            using T = core::value_type_t<in_mds_t>;

            if constexpr (in_mds_t::rank() == 0) {
                constexpr auto data =
                    detail::uniform_distribution<T, 1>(0, 1)[0];
                in_mds() = data;

            } else {
                constexpr auto data_size =
                    []<size_t... Is>(std::index_sequence<Is...>) {
                        return (in_mds_t::static_extent(Is) * ...);
                    }(std::make_index_sequence<in_mds_t::rank()>{});

                constexpr auto data =
                    core::container<T, typename in_mds_t::extents_type>{
                        detail::uniform_distribution<T, data_size>(0, 1)};

                copy_to(data, in_mds);
            }

            return;
        }
    }

    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::rand_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true>{}, in_mds);
}

template <std::floating_point dtype = float,
          core::MPMode mpmode = core::MPMode::NONE,
          core::extents_c exts_t = core::extents<uint8_t>>
[[nodiscard]] inline constexpr auto rand(exts_t &&exts = exts_t{}) noexcept {
    auto out = empty<dtype>(std::forward<exts_t>(exts));
    rand_to<mpmode>(out);
    return out;
}

} // namespace random
} // namespace mdtensor
