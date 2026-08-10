/**
 * @file
 * @brief test
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#include <gtest/gtest.h>

#ifdef MDTENSOR_USE_SINGLE_HEADER // for single header include
#include "mdtensor.hpp"
#else
#include "mdtensor/mdtensor.hpp"
#endif

namespace md = mdtensor;

template <typename... Ts>
using test_t = typename md::core::common_data_type_t<Ts...>;

template <typename... Ts>
constexpr bool assigned = requires { typename test_t<Ts...>; };

TEST(test, assign) {
    static_assert(assigned<std::uint8_t>);
    static_assert(assigned<std::uint16_t>);
    static_assert(assigned<std::uint32_t>);
    static_assert(assigned<uint64_t>);
    static_assert(assigned<std::int8_t>);
    static_assert(assigned<std::int16_t>);
    static_assert(assigned<std::int32_t>);
    static_assert(assigned<std::int64_t>);
    static_assert(assigned<float>);
    static_assert(assigned<double>);
    static_assert(assigned<long double>);
}

TEST(test, not_assign) {
    static_assert(!assigned<void>);
    static_assert(!assigned<std::nullopt_t>);
    static_assert(!assigned<std::optional<int>>);
    static_assert(!assigned<std::string>);
    static_assert(!assigned<std::vector<int>>);
    static_assert(!assigned<std::array<int, 3>>);
    static_assert(!assigned<std::tuple<int, double>>);
    static_assert(!assigned<std::pair<int, double>>);
}

TEST(test, uint_with_uint) {
    static_assert(
        std::same_as<test_t<std::uint8_t, std::uint8_t>, std::uint8_t>);
    static_assert(
        std::same_as<test_t<std::uint8_t, std::uint16_t>, std::uint16_t>);
    static_assert(
        std::same_as<test_t<std::uint8_t, std::uint32_t>, std::uint32_t>);
    static_assert(std::same_as<test_t<std::uint8_t, uint64_t>, uint64_t>);

    static_assert(
        std::same_as<test_t<std::uint16_t, std::uint8_t>, std::uint16_t>);
    static_assert(
        std::same_as<test_t<std::uint16_t, std::uint16_t>, std::uint16_t>);
    static_assert(
        std::same_as<test_t<std::uint16_t, std::uint32_t>, std::uint32_t>);
    static_assert(std::same_as<test_t<std::uint16_t, uint64_t>, uint64_t>);

    static_assert(
        std::same_as<test_t<std::uint32_t, std::uint8_t>, std::uint32_t>);
    static_assert(
        std::same_as<test_t<std::uint32_t, std::uint16_t>, std::uint32_t>);
    static_assert(
        std::same_as<test_t<std::uint32_t, std::uint32_t>, std::uint32_t>);
    static_assert(std::same_as<test_t<std::uint32_t, uint64_t>, uint64_t>);

    static_assert(std::same_as<test_t<uint64_t, std::uint8_t>, uint64_t>);
    static_assert(std::same_as<test_t<uint64_t, std::uint16_t>, uint64_t>);
    static_assert(std::same_as<test_t<uint64_t, std::uint32_t>, uint64_t>);
    static_assert(std::same_as<test_t<uint64_t, uint64_t>, uint64_t>);
}

TEST(test, int_with_int) {
    static_assert(std::same_as<test_t<std::int8_t, std::int8_t>, std::int8_t>);
    static_assert(
        std::same_as<test_t<std::int8_t, std::int16_t>, std::int16_t>);
    static_assert(
        std::same_as<test_t<std::int8_t, std::int32_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::int8_t, std::int64_t>, std::int64_t>);

    static_assert(
        std::same_as<test_t<std::int16_t, std::int8_t>, std::int16_t>);
    static_assert(
        std::same_as<test_t<std::int16_t, std::int16_t>, std::int16_t>);
    static_assert(
        std::same_as<test_t<std::int16_t, std::int32_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::int16_t, std::int64_t>, std::int64_t>);

    static_assert(
        std::same_as<test_t<std::int32_t, std::int8_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::int32_t, std::int16_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::int32_t, std::int32_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::int32_t, std::int64_t>, std::int64_t>);

    static_assert(
        std::same_as<test_t<std::int64_t, std::int8_t>, std::int64_t>);
    static_assert(
        std::same_as<test_t<std::int64_t, std::int16_t>, std::int64_t>);
    static_assert(
        std::same_as<test_t<std::int64_t, std::int32_t>, std::int64_t>);
    static_assert(
        std::same_as<test_t<std::int64_t, std::int64_t>, std::int64_t>);
}

TEST(test, uint_with_int) {
    static_assert(
        std::same_as<test_t<std::uint8_t, std::int8_t>, std::int16_t>);
    static_assert(
        std::same_as<test_t<std::uint8_t, std::int16_t>, std::int16_t>);
    static_assert(
        std::same_as<test_t<std::uint8_t, std::int32_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::uint8_t, std::int64_t>, std::int64_t>);

    static_assert(
        std::same_as<test_t<std::uint16_t, std::int8_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::uint16_t, std::int16_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::uint16_t, std::int32_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::uint16_t, std::int64_t>, std::int64_t>);

    static_assert(
        std::same_as<test_t<std::uint32_t, std::int8_t>, std::int64_t>);
    static_assert(
        std::same_as<test_t<std::uint32_t, std::int16_t>, std::int64_t>);
    static_assert(
        std::same_as<test_t<std::uint32_t, std::int32_t>, std::int64_t>);
    static_assert(
        std::same_as<test_t<std::uint32_t, std::int64_t>, std::int64_t>);

    static_assert(!assigned<uint64_t, std::int8_t>);
    static_assert(!assigned<uint64_t, std::int16_t>);
    static_assert(!assigned<uint64_t, std::int32_t>);
    static_assert(!assigned<uint64_t, std::int64_t>);
}

TEST(test, int_with_uint) {
    static_assert(
        std::same_as<test_t<std::int8_t, std::uint8_t>, std::int16_t>);
    static_assert(
        std::same_as<test_t<std::int8_t, std::uint16_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::int8_t, std::uint32_t>, std::int64_t>);
    static_assert(!assigned<std::int8_t, uint64_t>);

    static_assert(
        std::same_as<test_t<std::int16_t, std::uint8_t>, std::int16_t>);
    static_assert(
        std::same_as<test_t<std::int16_t, std::uint16_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::int16_t, std::uint32_t>, std::int64_t>);
    static_assert(!assigned<std::int16_t, uint64_t>);

    static_assert(
        std::same_as<test_t<std::int32_t, std::uint8_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::int32_t, std::uint16_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::int32_t, std::uint32_t>, std::int64_t>);
    static_assert(!assigned<std::int32_t, uint64_t>);

    static_assert(
        std::same_as<test_t<std::int64_t, std::uint8_t>, std::int64_t>);
    static_assert(
        std::same_as<test_t<std::int64_t, std::uint16_t>, std::int64_t>);
    static_assert(
        std::same_as<test_t<std::int64_t, std::uint32_t>, std::int64_t>);
    static_assert(!assigned<std::int64_t, uint64_t>);
}

TEST(test, fpoint_with_fpoint) {
    static_assert(std::same_as<test_t<float, float>, float>);
    static_assert(std::same_as<test_t<float, double>, double>);
    static_assert(std::same_as<test_t<double, float>, double>);
    static_assert(std::same_as<test_t<double, double>, double>);
}

TEST(test, fpoint_with_uint) {
    static_assert(std::same_as<test_t<float, std::uint8_t>, float>);
    static_assert(std::same_as<test_t<float, std::uint16_t>, float>);
    static_assert(std::same_as<test_t<float, std::uint32_t>, float>);
    static_assert(std::same_as<test_t<float, uint64_t>, float>);

    static_assert(std::same_as<test_t<double, std::uint8_t>, double>);
    static_assert(std::same_as<test_t<double, std::uint16_t>, double>);
    static_assert(std::same_as<test_t<double, std::uint32_t>, double>);
    static_assert(std::same_as<test_t<double, uint64_t>, double>);
}

TEST(test, fpoint_with_int) {
    static_assert(std::same_as<test_t<float, std::int8_t>, float>);
    static_assert(std::same_as<test_t<float, std::int16_t>, float>);
    static_assert(std::same_as<test_t<float, std::int32_t>, float>);
    static_assert(std::same_as<test_t<float, std::int64_t>, float>);

    static_assert(std::same_as<test_t<double, std::int8_t>, double>);
    static_assert(std::same_as<test_t<double, std::int16_t>, double>);
    static_assert(std::same_as<test_t<double, std::int32_t>, double>);
    static_assert(std::same_as<test_t<double, std::int64_t>, double>);
}

TEST(test, bool_with_bool) {
    static_assert(std::same_as<test_t<bool, bool>, bool>);
}

TEST(test, bool_with_uint) {
    static_assert(std::same_as<test_t<bool, std::uint8_t>, std::uint8_t>);
    static_assert(std::same_as<test_t<bool, std::uint16_t>, std::uint16_t>);
    static_assert(std::same_as<test_t<bool, std::uint32_t>, std::uint32_t>);
    static_assert(std::same_as<test_t<bool, uint64_t>, uint64_t>);
}

TEST(test, bool_with_int) {
    static_assert(std::same_as<test_t<bool, std::int8_t>, std::int8_t>);
    static_assert(std::same_as<test_t<bool, std::int16_t>, std::int16_t>);
    static_assert(std::same_as<test_t<bool, std::int32_t>, std::int32_t>);
    static_assert(std::same_as<test_t<bool, std::int64_t>, std::int64_t>);
}

TEST(test, bool_with_fpoint) {
    static_assert(std::same_as<test_t<bool, float>, float>);
    static_assert(std::same_as<test_t<bool, double>, double>);
}

TEST(test, triple) {
    static_assert(std::same_as<test_t<float, float, float>, float>);
}
