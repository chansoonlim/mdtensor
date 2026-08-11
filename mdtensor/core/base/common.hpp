/**
 * @file
 * @brief Common utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "concept.hpp"

namespace mdtensor::core {
namespace detail {

template <std::size_t Size> struct signed_by_size;

template <> struct signed_by_size<1> {
    using type = std::int8_t;
};

template <> struct signed_by_size<2> {
    using type = std::int16_t;
};

template <> struct signed_by_size<4> {
    using type = std::int32_t;
};

template <> struct signed_by_size<8> {
    using type = std::int64_t;
};

template <std::size_t Size>
using signed_by_size_t = typename signed_by_size<Size>::type;

template <typename T>
constexpr bool valid_extent_index_v =
    integral_c<T> && !std::same_as<std::remove_cvref_t<T>, bool>;

template <typename... Ts> struct common_integer_type_impl {
    // no type
};

template <typename T>
    requires valid_extent_index_v<T>
struct common_integer_type_impl<T> {
    using type = std::remove_cvref_t<T>;
};

template <typename T1, typename T2>
    requires(valid_extent_index_v<T1> && valid_extent_index_v<T2> &&
             (std::is_signed_v<std::remove_cvref_t<T1>> ==
              std::is_signed_v<std::remove_cvref_t<T2>>))
struct common_integer_type_impl<T1, T2> {
    using type =
        std::conditional_t<(sizeof(std::remove_cvref_t<T1>) >=
                            sizeof(std::remove_cvref_t<T2>)),
                           std::remove_cvref_t<T1>, std::remove_cvref_t<T2>>;
};

template <typename S, typename U>
    requires(valid_extent_index_v<S> && valid_extent_index_v<U> &&
             std::is_signed_v<std::remove_cvref_t<S>> &&
             std::is_unsigned_v<std::remove_cvref_t<U>> &&
             (sizeof(std::remove_cvref_t<U>) < 8))
struct common_integer_type_impl<S, U> {
  private:
    static constexpr std::size_t size =
        (sizeof(std::remove_cvref_t<S>) > sizeof(std::remove_cvref_t<U>))
            ? sizeof(std::remove_cvref_t<S>)
            : sizeof(std::remove_cvref_t<U>) * 2;

  public:
    using type = signed_by_size_t<size>;
};

template <typename U, typename S>
    requires(valid_extent_index_v<U> && valid_extent_index_v<S> &&
             std::is_unsigned_v<std::remove_cvref_t<U>> &&
             std::is_signed_v<std::remove_cvref_t<S>> &&
             (sizeof(std::remove_cvref_t<U>) < 8))
struct common_integer_type_impl<U, S>
    : common_integer_type_impl<std::remove_cvref_t<S>, std::remove_cvref_t<U>> {
};

template <typename T1, typename T2, typename... Ts>
    requires(sizeof...(Ts) > 0 &&
             requires { typename common_integer_type_impl<T1, T2>::type; })
struct common_integer_type_impl<T1, T2, Ts...> {
  public:
    using type = typename common_integer_type_impl<
        typename common_integer_type_impl<T1, T2>::type, Ts...>::type;
};

} // namespace detail

template <typename... Ts>
using common_integer_type_t =
    typename detail::common_integer_type_impl<Ts...>::type;

namespace detail {

template <typename T>
constexpr bool data_bool_v = std::same_as<std::remove_cvref_t<T>, bool>;

template <typename T> constexpr bool data_fpoint_v = floating_point_c<T>;

template <typename T>
constexpr bool data_integer_v =
    std::integral<std::remove_cvref_t<T>> && !data_bool_v<T>;

template <typename T>
constexpr bool data_scalar_v =
    data_bool_v<T> || data_integer_v<T> || data_fpoint_v<T>;

template <typename T1, typename T2> struct common_data_pair_impl {
    // no type
};

// bool + bool -> bool
template <typename T1, typename T2>
    requires(data_bool_v<T1> && data_bool_v<T2>)
struct common_data_pair_impl<T1, T2> {
    using type = bool;
};

// bool + T -> T
template <typename B, typename T>
    requires(data_bool_v<B> && !data_bool_v<T> && data_scalar_v<T>)
struct common_data_pair_impl<B, T> {
    using type = std::remove_cvref_t<T>;
};

// T + bool -> T
template <typename T, typename B>
    requires(!data_bool_v<T> && data_scalar_v<T> && data_bool_v<B>)
struct common_data_pair_impl<T, B> {
    using type = std::remove_cvref_t<T>;
};

// floating + scalar -> floating
template <typename T1, typename T2>
    requires(!data_bool_v<T1> && !data_bool_v<T2> && data_scalar_v<T1> &&
             data_scalar_v<T2> && (data_fpoint_v<T1> || data_fpoint_v<T2>))
struct common_data_pair_impl<T1, T2> {
  private:
    using lhs_t = std::remove_cvref_t<T1>;
    using rhs_t = std::remove_cvref_t<T2>;

  public:
    using type = std::conditional_t<
        data_fpoint_v<T1> && data_fpoint_v<T2>,
        std::conditional_t<(sizeof(lhs_t) >= sizeof(rhs_t)), lhs_t, rhs_t>,
        std::conditional_t<data_fpoint_v<T1>, lhs_t, rhs_t>>;
};

// integer + integer -> common_integer_type_t
template <typename T1, typename T2>
    requires(data_integer_v<T1> && data_integer_v<T2> &&
             requires {
                 typename common_integer_type_t<std::remove_cvref_t<T1>,
                                                std::remove_cvref_t<T2>>;
             })
struct common_data_pair_impl<T1, T2> {
    using type =
        common_integer_type_t<std::remove_cvref_t<T1>, std::remove_cvref_t<T2>>;
};

template <typename... Ts> struct data_promote_impl {
    // no type
};

template <typename T>
    requires data_scalar_v<T>
struct data_promote_impl<T> {
    using type = std::remove_cvref_t<T>;
};

template <typename T1, typename T2>
struct data_promote_impl<T1, T2>
    : common_data_pair_impl<std::remove_cvref_t<T1>, std::remove_cvref_t<T2>> {
};

template <typename T1, typename T2, typename... Ts>
    requires requires {
        typename common_data_pair_impl<std::remove_cvref_t<T1>,
                                       std::remove_cvref_t<T2>>::type;
    }
struct data_promote_impl<T1, T2, Ts...> {
  private:
    using pair_t =
        typename common_data_pair_impl<std::remove_cvref_t<T1>,
                                       std::remove_cvref_t<T2>>::type;

  public:
    using type = typename data_promote_impl<pair_t, Ts...>::type;
};

} // namespace detail

template <typename... Ts>
    requires(sizeof...(Ts) > 0 && (detail::data_scalar_v<Ts> && ...))
using common_data_type_t = typename detail::data_promote_impl<Ts...>::type;

template <std::integral index_t>
[[nodiscard]] constexpr index_t bounding_index(index_t index,
                                               const std::size_t &bound) {
    if constexpr (std::is_signed_v<index_t>) {
        if (index < index_t{0}) {
            index = static_cast<index_t>(bound + 1 -
                                         static_cast<std::size_t>(-index));
        }
    }

    if (index < index_t{0} || bound < static_cast<std::size_t>(index)) {
        throw std::out_of_range(
            "Index is out of bounds: " + std::to_string(index) +
            " is not in [0, " + std::to_string(bound) + ").");
    }

    return index;
}

template <std::integral in_t, in_t... ins, typename compare_t>
[[nodiscard]] consteval auto
get_sorted_array(std::integer_sequence<in_t, ins...>,
                 compare_t compare) noexcept {
    auto arr = std::array{ins...};
    std::sort(arr.begin(), arr.end(), compare);
    return arr;
}

template <std::size_t rank, std::integral axes_t, axes_t... axes,
          typename compare_t>
[[nodiscard]] consteval auto
get_sorted_axes(std::integer_sequence<axes_t, axes...>,
                compare_t compare) noexcept {
    constexpr auto arr =
        get_sorted_array(std::index_sequence<static_cast<std::size_t>(
                             bounding_index<axes_t>(axes, rank - 1))...>{},
                         compare);

    if constexpr (1 < arr.size()) {
        static_assert(
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return ((arr[Is] != arr[Is + 1]) && ...);
            }(std::make_index_sequence<arr.size() - 1>{}),
            "Duplicate axes are not allowed.");
    }

    return arr;
}

template <typename value_t, std::size_t size>
[[nodiscard]] constexpr bool contains(const std::array<value_t, size> &array,
                                      const value_t &value) noexcept {
    for (const auto element : array) {
        if (element == value) {
            return true;
        }
    }

    return false;
}

} // namespace mdtensor::core
