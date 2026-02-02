/**
 * @file
 * @brief String conversion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace detail {

template <typename in_t>
[[nodiscard]] inline std::string to_string_impl(in_t &&in) {
    using index_t = typename std::remove_cvref_t<in_t>::index_type;

    std::string str = "[";

    if constexpr (in_t::rank() == 0) {
        return std::to_string(in());

    } else {
        for (index_t i = 0; i < in.extent(0); i++) {
            str += to_string_impl(core::submdspan_from_left(in, i));

            if (i < in.extent(0) - 1) {
                str += ", ";
            }
        }
    }

    return str + "]";
}

} // namespace detail

/**
 * @brief Convert an extents object into a shape string.
 *
 * @tparam in_t Extents type satisfying extents_c.
 *
 * @param in Input extents.
 *
 * @return Shape string in the form "(d0, d1, ...)".
 *
 * @note This overload is useful for printing tensor shapes.
 */
template <extents_c exts_t>
[[nodiscard]] inline std::string to_string(exts_t &&exts) {
    using exts_base_t = std::remove_cvref_t<exts_t>;

    std::string str = "(";

    for (size_t i = 0; i < exts_base_t::rank(); i++) {
        str += std::to_string(exts.extent(i));
        if (i < exts_base_t::rank() - 1) {
            str += ", ";
        }
    }

    return str + ")";
}

/**
 * @brief Convert a tensor-like object into a nested bracket string.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return Nested bracket string representation.
 *
 * @note The input is converted to a const mdspan view before formatting.
 *
 * @see mdtensor::to_string(const extents_c&) for shape formatting.
 */
template <typename in_t> [[nodiscard]] inline std::string to_string(in_t &&in) {
    return detail::to_string_impl(
        core::to_const_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
