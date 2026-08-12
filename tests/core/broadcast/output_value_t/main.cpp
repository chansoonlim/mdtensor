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

template <typename dtype, typename... Ts>
using test_t = typename md::core::output_value_t<dtype, Ts...>;

template <typename dtype, typename... Ts>
constexpr bool assigned = requires { typename test_t<dtype, Ts...>; };

TEST(test, assign) {
    static_assert(assigned<void, std::uint8_t>);
    static_assert(assigned<void, std::uint16_t>);
    static_assert(assigned<void, std::uint32_t>);
    static_assert(assigned<void, std::uint64_t>);
    static_assert(assigned<void, std::int8_t>);
    static_assert(assigned<void, std::int16_t>);
    static_assert(assigned<void, std::int32_t>);
    static_assert(assigned<void, std::int64_t>);
    static_assert(assigned<void, float>);
    static_assert(assigned<void, double>);
    static_assert(assigned<void, long double>);

    static_assert(assigned<int, std::nullopt_t>);
    static_assert(assigned<int, std::optional<int>>);
    static_assert(assigned<int, std::string>);
    static_assert(assigned<int, std::vector<int>>);
    static_assert(assigned<int, std::array<int, 3>>);
    static_assert(assigned<int, std::tuple<int, double>>);
    static_assert(assigned<int, std::pair<int, double>>);
}

TEST(test, define_dtype) {
    static_assert(std::same_as<test_t<int, std::uint8_t>, int>);
    static_assert(std::same_as<test_t<int, std::uint16_t>, int>);
    static_assert(std::same_as<test_t<int, std::uint32_t>, int>);
    static_assert(std::same_as<test_t<int, std::uint64_t>, int>);
    static_assert(std::same_as<test_t<int, std::int8_t>, int>);
    static_assert(std::same_as<test_t<int, std::int16_t>, int>);
    static_assert(std::same_as<test_t<int, std::int32_t>, int>);
    static_assert(std::same_as<test_t<int, std::int64_t>, int>);
    static_assert(std::same_as<test_t<int, float>, int>);
    static_assert(std::same_as<test_t<int, double>, int>);
    static_assert(std::same_as<test_t<int, long double>, int>);

    static_assert(std::same_as<test_t<int, std::nullopt_t>, int>);
    static_assert(std::same_as<test_t<int, std::optional<int>>, int>);
    static_assert(std::same_as<test_t<int, std::string>, int>);
    static_assert(std::same_as<test_t<int, std::vector<int>>, int>);
    static_assert(std::same_as<test_t<int, std::array<int, 3>>, int>);
    static_assert(std::same_as<test_t<int, std::tuple<int, double>>, int>);
    static_assert(std::same_as<test_t<int, std::pair<int, double>>, int>);
}

TEST(test, undefine_dtype) {
    static_assert(std::same_as<test_t<void, std::uint8_t>, std::uint8_t>);
    static_assert(std::same_as<test_t<void, std::uint16_t>, std::uint16_t>);
    static_assert(std::same_as<test_t<void, std::uint32_t>, std::uint32_t>);
    static_assert(std::same_as<test_t<void, std::uint64_t>, std::uint64_t>);
    static_assert(std::same_as<test_t<void, std::int8_t>, std::int8_t>);
    static_assert(std::same_as<test_t<void, std::int16_t>, std::int16_t>);
    static_assert(std::same_as<test_t<void, std::int32_t>, std::int32_t>);
    static_assert(std::same_as<test_t<void, std::int64_t>, std::int64_t>);
    static_assert(std::same_as<test_t<void, float>, float>);
    static_assert(std::same_as<test_t<void, double>, double>);
    static_assert(std::same_as<test_t<void, long double>, long double>);
}

TEST(test, 1) {
    using value_t = int;
    using index_t = std::size_t;

    const value_t a = 1;

    static_assert(
        std::same_as<test_t<void, decltype(md::core::to_mdspan(a))>, value_t>);
    static_assert(
        std::same_as<test_t<void, decltype(md::core::to_const_mdspan(a))>,
                     value_t>);
}
