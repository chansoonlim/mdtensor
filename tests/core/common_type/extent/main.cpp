#include <gtest/gtest.h>

#include "mdtensor/core/broadcast.hpp"

namespace md = mdtensor;

template <typename... Ts>
using test_t = typename md::core::extent_common_type_t<Ts...>;

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
