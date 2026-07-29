/**
 * @file
 * @brief SplitMix64 random number generator for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <cstdint>

namespace mdtensor::random::generator {

class SplitMix64 {
  public:
    using result_type = std::uint64_t;

  public:
    constexpr explicit SplitMix64(result_type seed) noexcept : state_(seed) {}

  public:
    [[nodiscard]] constexpr result_type operator()() noexcept {
        // https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64

        result_type z = (state_ += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;

        return z ^ (z >> 31);
    }

  private:
    result_type state_;
};

} // namespace mdtensor::random::generator
