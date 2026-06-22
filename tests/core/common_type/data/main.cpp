#include <gtest/gtest.h>

#include "mdtensor/core/broadcast.hpp"

namespace md = mdtensor;

template <typename... Ts>
using test_t = typename md::core::common_data_type_t<Ts...>;

template <typename... Ts>
constexpr bool assigned = requires { typename test_t<Ts...>; };

TEST(test, uint_with_uint) {
    static_assert(std::same_as<test_t<uint8_t, uint8_t>, uint8_t>);
    static_assert(std::same_as<test_t<uint8_t, uint16_t>, uint16_t>);
    static_assert(std::same_as<test_t<uint8_t, uint32_t>, uint32_t>);
    static_assert(std::same_as<test_t<uint8_t, uint64_t>, uint64_t>);

    static_assert(std::same_as<test_t<uint16_t, uint8_t>, uint16_t>);
    static_assert(std::same_as<test_t<uint16_t, uint16_t>, uint16_t>);
    static_assert(std::same_as<test_t<uint16_t, uint32_t>, uint32_t>);
    static_assert(std::same_as<test_t<uint16_t, uint64_t>, uint64_t>);

    static_assert(std::same_as<test_t<uint32_t, uint8_t>, uint32_t>);
    static_assert(std::same_as<test_t<uint32_t, uint16_t>, uint32_t>);
    static_assert(std::same_as<test_t<uint32_t, uint32_t>, uint32_t>);
    static_assert(std::same_as<test_t<uint32_t, uint64_t>, uint64_t>);

    static_assert(std::same_as<test_t<uint64_t, uint8_t>, uint64_t>);
    static_assert(std::same_as<test_t<uint64_t, uint16_t>, uint64_t>);
    static_assert(std::same_as<test_t<uint64_t, uint32_t>, uint64_t>);
    static_assert(std::same_as<test_t<uint64_t, uint64_t>, uint64_t>);
}

TEST(test, int_with_int) {
    static_assert(std::same_as<test_t<int8_t, int8_t>, int8_t>);
    static_assert(std::same_as<test_t<int8_t, int16_t>, int16_t>);
    static_assert(std::same_as<test_t<int8_t, int32_t>, int32_t>);
    static_assert(std::same_as<test_t<int8_t, int64_t>, int64_t>);

    static_assert(std::same_as<test_t<int16_t, int8_t>, int16_t>);
    static_assert(std::same_as<test_t<int16_t, int16_t>, int16_t>);
    static_assert(std::same_as<test_t<int16_t, int32_t>, int32_t>);
    static_assert(std::same_as<test_t<int16_t, int64_t>, int64_t>);

    static_assert(std::same_as<test_t<int32_t, int8_t>, int32_t>);
    static_assert(std::same_as<test_t<int32_t, int16_t>, int32_t>);
    static_assert(std::same_as<test_t<int32_t, int32_t>, int32_t>);
    static_assert(std::same_as<test_t<int32_t, int64_t>, int64_t>);

    static_assert(std::same_as<test_t<int64_t, int8_t>, int64_t>);
    static_assert(std::same_as<test_t<int64_t, int16_t>, int64_t>);
    static_assert(std::same_as<test_t<int64_t, int32_t>, int64_t>);
    static_assert(std::same_as<test_t<int64_t, int64_t>, int64_t>);
}

TEST(test, uint_with_int) {
    static_assert(std::same_as<test_t<uint8_t, int8_t>, int16_t>);
    static_assert(std::same_as<test_t<uint8_t, int16_t>, int16_t>);
    static_assert(std::same_as<test_t<uint8_t, int32_t>, int32_t>);
    static_assert(std::same_as<test_t<uint8_t, int64_t>, int64_t>);

    static_assert(std::same_as<test_t<uint16_t, int8_t>, int32_t>);
    static_assert(std::same_as<test_t<uint16_t, int16_t>, int32_t>);
    static_assert(std::same_as<test_t<uint16_t, int32_t>, int32_t>);
    static_assert(std::same_as<test_t<uint16_t, int64_t>, int64_t>);

    static_assert(std::same_as<test_t<uint32_t, int8_t>, int64_t>);
    static_assert(std::same_as<test_t<uint32_t, int16_t>, int64_t>);
    static_assert(std::same_as<test_t<uint32_t, int32_t>, int64_t>);
    static_assert(std::same_as<test_t<uint32_t, int64_t>, int64_t>);

    static_assert(!assigned<uint64_t, int8_t>);
    static_assert(!assigned<uint64_t, int16_t>);
    static_assert(!assigned<uint64_t, int32_t>);
    static_assert(!assigned<uint64_t, int64_t>);
}

TEST(test, int_with_uint) {
    static_assert(std::same_as<test_t<int8_t, uint8_t>, int16_t>);
    static_assert(std::same_as<test_t<int8_t, uint16_t>, int32_t>);
    static_assert(std::same_as<test_t<int8_t, uint32_t>, int64_t>);
    static_assert(!assigned<int8_t, uint64_t>);

    static_assert(std::same_as<test_t<int16_t, uint8_t>, int16_t>);
    static_assert(std::same_as<test_t<int16_t, uint16_t>, int32_t>);
    static_assert(std::same_as<test_t<int16_t, uint32_t>, int64_t>);
    static_assert(!assigned<int16_t, uint64_t>);

    static_assert(std::same_as<test_t<int32_t, uint8_t>, int32_t>);
    static_assert(std::same_as<test_t<int32_t, uint16_t>, int32_t>);
    static_assert(std::same_as<test_t<int32_t, uint32_t>, int64_t>);
    static_assert(!assigned<int32_t, uint64_t>);

    static_assert(std::same_as<test_t<int64_t, uint8_t>, int64_t>);
    static_assert(std::same_as<test_t<int64_t, uint16_t>, int64_t>);
    static_assert(std::same_as<test_t<int64_t, uint32_t>, int64_t>);
    static_assert(!assigned<int64_t, uint64_t>);
}

TEST(test, fpoint_with_fpoint) {
    static_assert(std::same_as<test_t<float, float>, float>);
    static_assert(std::same_as<test_t<float, double>, double>);
    static_assert(std::same_as<test_t<double, float>, double>);
    static_assert(std::same_as<test_t<double, double>, double>);
}

TEST(test, fpoint_with_uint) {
    static_assert(std::same_as<test_t<float, uint8_t>, float>);
    static_assert(std::same_as<test_t<float, uint16_t>, float>);
    static_assert(std::same_as<test_t<float, uint32_t>, float>);
    static_assert(std::same_as<test_t<float, uint64_t>, float>);

    static_assert(std::same_as<test_t<double, uint8_t>, double>);
    static_assert(std::same_as<test_t<double, uint16_t>, double>);
    static_assert(std::same_as<test_t<double, uint32_t>, double>);
    static_assert(std::same_as<test_t<double, uint64_t>, double>);
}

TEST(test, fpoint_with_int) {
    static_assert(std::same_as<test_t<float, int8_t>, float>);
    static_assert(std::same_as<test_t<float, int16_t>, float>);
    static_assert(std::same_as<test_t<float, int32_t>, float>);
    static_assert(std::same_as<test_t<float, int64_t>, float>);

    static_assert(std::same_as<test_t<double, int8_t>, double>);
    static_assert(std::same_as<test_t<double, int16_t>, double>);
    static_assert(std::same_as<test_t<double, int32_t>, double>);
    static_assert(std::same_as<test_t<double, int64_t>, double>);
}

TEST(test, bool_with_bool) {
    static_assert(std::same_as<test_t<bool, bool>, bool>);
}

TEST(test, bool_with_uint) {
    static_assert(std::same_as<test_t<bool, uint8_t>, uint8_t>);
    static_assert(std::same_as<test_t<bool, uint16_t>, uint16_t>);
    static_assert(std::same_as<test_t<bool, uint32_t>, uint32_t>);
    static_assert(std::same_as<test_t<bool, uint64_t>, uint64_t>);
}

TEST(test, bool_with_int) {
    static_assert(std::same_as<test_t<bool, int8_t>, int8_t>);
    static_assert(std::same_as<test_t<bool, int16_t>, int16_t>);
    static_assert(std::same_as<test_t<bool, int32_t>, int32_t>);
    static_assert(std::same_as<test_t<bool, int64_t>, int64_t>);
}

TEST(test, bool_with_fpoint) {
    static_assert(std::same_as<test_t<bool, float>, float>);
    static_assert(std::same_as<test_t<bool, double>, double>);
}
