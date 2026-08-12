/**
 * @file
 * @brief Type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

// TODO: include std headers at each file instead of this header
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "arithmetic.hpp"
#include "common_arithmetic_type.hpp"
#include "common_integral_type.hpp"
#include "concept.hpp"
#include "filtered_common_arithmetic_type.hpp"
#include "kokkos.hpp"
#include "null.hpp"
