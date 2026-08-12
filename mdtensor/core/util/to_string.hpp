/**
 * @file
 * @brief To-string conversion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <charconv>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include "../base/base.hpp"

namespace mdtensor::core {
namespace detail {

inline constexpr int numpy_default_precision = 8;

template <std::floating_point value_t>
[[nodiscard]] inline std::string floating_value_to_string(const value_t value) {
    if constexpr (requires {
                      { std::isnan(value) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(value)) {
            return "nan";
        }
    }

    if constexpr (requires {
                      { std::isinf(value) } -> std::convertible_to<bool>;
                  }) {
        if (std::isinf(value)) {
            return std::signbit(value) ? "-inf" : "inf";
        }
    }

    const value_t magnitude = std::abs(value);

    /*
     * NumPy-like scalar approximation:
     *
     * - fixed-point:      1e-4 <= |x| < 1e8, including zero
     * - scientific:       0 < |x| < 1e-4 or |x| >= 1e8
     *
     * NumPy's exact choice is array-wide, so mixed arrays may use
     * scientific notation even when an individual value would not.
     */
    const bool use_scientific =
        magnitude != value_t{0} &&
        (magnitude < value_t{1e-4} || magnitude >= value_t{1e8});

    const std::chars_format format = use_scientific
                                         ? std::chars_format::scientific
                                         : std::chars_format::fixed;

    std::array<char, 128> buffer{};

    // First obtain the shortest round-trippable representation.
    auto [ptr, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                   value, format);

    if (ec != std::errc{}) {
        throw std::runtime_error(
            "Failed to convert floating-point value to string.");
    }

    std::string text(buffer.data(), ptr);

    const auto fractional_digits = [](const std::string &str) -> std::size_t {
        const std::size_t exponent_pos = str.find_first_of("eE");

        const std::size_t mantissa_end =
            exponent_pos == std::string::npos ? str.size() : exponent_pos;

        const std::size_t decimal_pos = str.find('.');

        if (decimal_pos == std::string::npos || decimal_pos >= mantissa_end) {
            return 0;
        }

        return mantissa_end - decimal_pos - 1;
    };

    // NumPy default: at most 8 fractional digits.
    if (fractional_digits(text) >
        static_cast<std::size_t>(numpy_default_precision)) {
        auto [rounded_ptr, rounded_ec] =
            std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                          format, numpy_default_precision);

        if (rounded_ec != std::errc{}) {
            throw std::runtime_error(
                "Failed to convert floating-point value to string.");
        }

        text.assign(buffer.data(), rounded_ptr);
    }

    // Remove unnecessary trailing zeros, but preserve the decimal point.
    const std::size_t exponent_pos = text.find_first_of("eE");

    const std::size_t mantissa_end =
        exponent_pos == std::string::npos ? text.size() : exponent_pos;

    const std::size_t decimal_pos = text.find('.');

    if (decimal_pos == std::string::npos || decimal_pos >= mantissa_end) {
        // "21"     -> "21."
        // "1e+08"  -> "1.e+08"
        text.insert(mantissa_end, 1, '.');

    } else {
        std::size_t trim_end = mantissa_end;

        while (trim_end > decimal_pos + 1 && text[trim_end - 1] == '0') {
            --trim_end;
        }

        // Decimal point itself is intentionally retained.
        text.erase(trim_end, mantissa_end - trim_end);
    }

    return text;
}

template <core::arithmetic_c value_t>
[[nodiscard]] inline std::string value_to_string(const value_t &value) {
    using base_t = std::remove_cvref_t<value_t>;

    if constexpr (std::same_as<base_t, bool>) {
        return value ? "true" : "false";

    } else if constexpr (std::is_integral_v<base_t>) {
        if constexpr (std::is_signed_v<base_t>) {
            return std::to_string(static_cast<long long>(value));

        } else {
            return std::to_string(static_cast<unsigned long long>(value));
        }

    } else if constexpr (std::is_floating_point_v<base_t>) {
        return floating_value_to_string(value);
    }
}

template <typename value_t>
[[nodiscard]] inline std::string
value_to_string(const std::optional<value_t> &value) {
    if (!value.has_value()) {
        return "nullopt";
    }

    return value_to_string(*value);
}

} // namespace detail

template <extents_c exts_t>
[[nodiscard]] inline std::string to_string(exts_t &&exts) {
    using base_t = std::remove_cvref_t<exts_t>;

    std::string str = "(";

    for (std::size_t i = 0; i < base_t::rank(); i++) {
        if (i != 0) {
            str += ", ";
        }

        str += std::to_string(exts.extent(i));
    }

    if constexpr (base_t::rank() == 1) {
        str += ",";
    }

    return str + ")";
}

template <typename in_t>
    requires(!extents_c<in_t>)
[[nodiscard]] inline std::string to_string(in_t &&in) {
    const auto in_mds = to_const_mdspan(std::forward<in_t>(in));

    std::string str = "[";

    if constexpr (in_mds.rank() == 0) {
        if constexpr (requires { detail::value_to_string(in_mds()); }) {
            return detail::value_to_string(in_mds());

        } else {
            return "NOT_STRING_CONVERTIBLE";
        }

    } else {
        using index_t = typename decltype(in_mds)::index_type;

        for (index_t i = 0; i < in_mds.extent(0); i++) {
            if (i != 0) {
                str += ", ";
            }

            str += to_string(submdspan_from_left(in_mds, i));
        }
    }

    return str + "]";
}

} // namespace mdtensor::core
