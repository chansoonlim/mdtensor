/**
 * @file
 * @brief Expect type traits for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "mdspan.hpp"

namespace mdtensor::core {

template <typename T>
using to_mdspan_t = decltype(to_mdspan(std::declval<T>()));

template <typename... Ts>
using common_value_type_t = core::common_arithmetic_type_t<
    typename std::remove_cvref_t<to_mdspan_t<Ts>>::value_type...>;

} // namespace mdtensor::core
