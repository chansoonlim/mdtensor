/**
 * @file
 * @brief Extents utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "common.hpp"
#include "type.hpp"

namespace mdtensor::core {

template <typename IndexType, std::size_t... Extents>
using extents = stdex::extents<IndexType, Extents...>;

template <typename IndexType, std::size_t Rank>
using dextents = stdex::dextents<IndexType, Rank>;

// NOTE: dims will be included in C++23
// (https://en.cppreference.com/w/cpp/container/mdspan/extents)
template <std::size_t Rank, class IndexType = std::size_t>
using dims = dextents<IndexType, Rank>;

namespace detail {

template <typename T> struct is_extents_impl : std::false_type {};

template <typename IndexType, std::size_t... Extents>
struct is_extents_impl<extents<IndexType, Extents...>> : std::true_type {};

} // namespace detail

// NOTE: stdex::detail::__is_extents is not used for godbolt test compatibility
template <typename T> struct is_extents : detail::is_extents_impl<T> {};

template <typename T> constexpr bool is_extents_v = is_extents<T>::value;

template <typename T>
concept extents_c = is_extents_v<std::remove_cvref_t<T>>;

constexpr auto dynamic_extent = stdex::dynamic_extent;
constexpr auto dyn = dynamic_extent;

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

template <typename... Ts> struct common_index_type_impl {
    // no type
};

template <typename T>
    requires valid_extent_index_v<T>
struct common_index_type_impl<T> {
    using type = std::remove_cvref_t<T>;
};

template <typename T1, typename T2>
    requires(valid_extent_index_v<T1> && valid_extent_index_v<T2> &&
             (std::is_signed_v<std::remove_cvref_t<T1>> ==
              std::is_signed_v<std::remove_cvref_t<T2>>))
struct common_index_type_impl<T1, T2> {
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
struct common_index_type_impl<S, U> {
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
struct common_index_type_impl<U, S>
    : common_index_type_impl<std::remove_cvref_t<S>, std::remove_cvref_t<U>> {};

template <typename T1, typename T2, typename... Ts>
    requires(sizeof...(Ts) > 0 &&
             requires { typename common_index_type_impl<T1, T2>::type; })
struct common_index_type_impl<T1, T2, Ts...> {
  public:
    using type = typename common_index_type_impl<
        typename common_index_type_impl<T1, T2>::type, Ts...>::type;
};

} // namespace detail

template <typename... Ts>
using common_index_type_t =
    typename detail::common_index_type_impl<Ts...>::type;

[[nodiscard]] constexpr auto to_extents(auto &&shape) {
    using base_t = std::remove_cvref_t<decltype(shape)>;

    if constexpr (extents_c<base_t>) {
        // If the input is already an extents, just return it as-is
        return std::forward<decltype(shape)>(shape);

    } else if constexpr (integral_c<base_t>) {
        if (shape < base_t{0}) {
            throw std::invalid_argument("shape must be non-negative");
        }

        using index_t = std::make_unsigned_t<base_t>;

        return dextents<index_t, 1>{static_cast<index_t>(shape)};
    }
}

template <extents_c in_t>
[[nodiscard]] constexpr std::size_t extents_size(in_t &&in) noexcept {
    if constexpr (in.rank() == 0) {
        // NOTE: mdspan with rank 0 can capture a single element.
        return 1;

    } else if constexpr (in.rank_dynamic() == 0) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return (in.static_extent(Is) * ...);
        }(std::make_index_sequence<in.rank()>{});

    } else {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return (static_cast<std::size_t>(in.extent(Is)) * ...);
        }(std::make_index_sequence<in.rank()>{});
    }
}

template <extents_c in_t>
[[nodiscard]] consteval bool is_always_same_extents() noexcept {
    return true;
}

template <extents_c in1_t, extents_c in2_t, extents_c... ins_t>
[[nodiscard]] consteval bool is_always_same_extents() noexcept {
    using base1_t = std::remove_cvref_t<in1_t>;
    using base2_t = std::remove_cvref_t<in2_t>;

    if constexpr (base1_t::rank() != base2_t::rank()) {
        return false;

    } else if constexpr (base1_t::rank_dynamic() != 0 ||
                         base2_t::rank_dynamic() != 0) {
        return false;

    } else if constexpr ([&]<std::size_t... Is>(std::index_sequence<Is...>) {
                             return ((base1_t::static_extent(Is) !=
                                      base2_t::static_extent(Is)) ||
                                     ...);
                         }(std::make_index_sequence<base1_t::rank()>{})) {
        return false;
    }

    if constexpr (sizeof...(ins_t) != 0) {
        return is_always_same_extents<in2_t, ins_t...>();

    } else {
        return true;
    }
}

template <extents_c in_t>
[[nodiscard]] constexpr bool is_same_extents(in_t &&in) noexcept {
    return true;
}

template <extents_c in1_t, extents_c in2_t, extents_c... ins_t>
[[nodiscard]] constexpr bool is_same_extents(in1_t &&in1, in2_t &&in2,
                                             ins_t &&...ins) noexcept {
    // NOTE: direct comparison of extents cannot catch rank difference.

    using base1_t = std::remove_cvref_t<in1_t>;
    using base2_t = std::remove_cvref_t<in2_t>;

    if constexpr (base1_t::rank() != base2_t::rank()) {
        return false;
    }

    using index_t = common_index_type_t<typename base1_t::index_type,
                                        typename base2_t::index_type>;

    for (std::size_t i = 0; i < base1_t::rank(); i++) {
        if (static_cast<index_t>(in1.extent(i)) !=
            static_cast<index_t>(in2.extent(i))) {
            return false;
        }
    }

    if constexpr (sizeof...(ins_t) != 0) {
        return is_same_extents(in2, ins...);

    } else {
        return true;
    }
}

template <std::size_t offset, std::size_t rank, extents_c in_t>
[[nodiscard]] constexpr auto slice_extents(in_t &&in) noexcept {
    using index_t = typename std::remove_cvref_t<in_t>::index_type;

    static_assert(in.rank() >= offset + rank,
                  "Incompatible offset and rank for slicing.");

    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return extents<index_t, in.static_extent(offset + Is)...>{
            in.extent(offset + Is)...};
    }(std::make_index_sequence<rank>{});
}

template <std::size_t rank, extents_c in_t>
[[nodiscard]] constexpr auto slice_extents_from_left(in_t &&in) noexcept {
    return slice_extents<0, rank>(std::forward<in_t>(in));
}

template <std::size_t rank, extents_c in_t>
[[nodiscard]] constexpr auto slice_extents_from_right(in_t &&in) noexcept {
    return slice_extents<std::remove_cvref_t<in_t>::rank() - rank, rank>(
        std::forward<in_t>(in));
}

template <extents_c in1_t, extents_c in2_t, extents_c... ins_t>
[[nodiscard]] constexpr auto compose_extents(in1_t &&in1, in2_t &&in2,
                                             ins_t &&...ins) noexcept {
    using base1_t = std::remove_cvref_t<in1_t>;
    using base2_t = std::remove_cvref_t<in2_t>;
    using index_t = common_index_type_t<typename base1_t::index_type,
                                        typename base2_t::index_type>;

    const auto cexts =
        [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>,
                                                  std::index_sequence<Js...>) {
            return extents<index_t, base1_t::static_extent(Is)...,
                           base2_t::static_extent(Js)...>{
                static_cast<index_t>(in1.extent(Is))...,
                static_cast<index_t>(in2.extent(Js))...};
        }(std::make_index_sequence<base1_t::rank()>{},
          std::make_index_sequence<base2_t::rank()>{});

    if constexpr (sizeof...(ins_t) == 0) {
        return cexts;

    } else {
        return compose_extents(cexts, std::forward<ins_t>(ins)...);
    }
}

namespace {

template <extents_c in_t>
[[nodiscard]] constexpr auto expand_extents_dims_impl_(in_t &&in,
                                                       std::index_sequence<>) {
    return std::forward<in_t>(in);
}

template <extents_c in_t, std::size_t axis, std::size_t... axes>
[[nodiscard]] constexpr auto
expand_extents_dims_impl_(in_t &&in, std::index_sequence<axis, axes...>) {
    // NOTE: this function requires ordered axes and does not check duplicates.

    using base_t = std::remove_cvref_t<in_t>;

    return expand_extents_dims_impl_(
        compose_extents(slice_extents_from_left<axis>(std::forward<in_t>(in)),
                        extents<uint8_t, 1>{},
                        slice_extents_from_right<base_t::rank() - axis>(
                            std::forward<in_t>(in))),
        std::index_sequence<axes...>{});
}

} // namespace

template <extents_c in_t, std::integral axes_t, axes_t... axes>
[[nodiscard]] constexpr auto
expand_extents_dims(in_t &&in, std::integer_sequence<axes_t, axes...>) {
    constexpr auto axes_sorted = get_sorted_axes<in.rank() + sizeof...(axes)>(
        std::integer_sequence<axes_t, axes...>{}, std::less<std::size_t>{});

    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return expand_extents_dims_impl_(
            std::forward<in_t>(in), std::index_sequence<axes_sorted[Is]...>{});
    }(std::make_index_sequence<axes_sorted.size()>{});
}

} // namespace mdtensor::core
