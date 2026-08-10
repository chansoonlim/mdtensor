/**
 * @file
 * @brief Random seed utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <random>

#include "generator/wrapper.hpp"

namespace mdtensor::random {

struct seed_t {
  public:
    using value_t = std::uint64_t;

  public:
    value_t value;
};

[[nodiscard]] inline seed_t make_random_seed() {
    auto engine = generator::EngineWrapper<std::random_device>{};

    return seed_t{engine.template get<typename seed_t::value_t>()};
}

} // namespace mdtensor::random
