/**
 * @file
 * @brief Type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <concepts>
#include <optional>
#include <tuple>

#ifndef MDSPAN_SINGLE_HEADER_INCLUDE_GUARD_ // NOTE: for godbolt test

// TODO: Remove when C++23 std::mdspan supports
#include <experimental/mdarray>
#include <experimental/mdspan>

#endif

namespace mdtensor {
namespace core {

#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
#define REAL_GCC __GNUC__ // probably
#endif

namespace stdex = std::experimental;

constexpr auto dyn = stdex::dynamic_extent;

template <typename IndexType, size_t... Extents>
using extents = stdex::extents<IndexType, Extents...>;

template <typename IndexType, size_t Rank>
using dextents = stdex::dextents<IndexType, Rank>;

// dims: will be included in C++23
// (https://en.cppreference.com/w/cpp/container/mdspan/extents)
template <size_t Rank, class IndexType = size_t>
using dims = stdex::dextents<IndexType, Rank>;

template <typename ElementType, typename Extents,
          typename LayoutPolicy = stdex::layout_right,
          typename AccessorPolicy = stdex::default_accessor<ElementType>>
using mdspan =
    stdex::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

constexpr auto full_extent = stdex::full_extent;

namespace detail {

template <typename T> struct is_extents_impl : std::false_type {};

template <typename IndexType, size_t... Extents>
struct is_extents_impl<stdex::extents<IndexType, Extents...>> : std::true_type {
};

} // namespace detail

template <typename T>
struct is_extents : detail::is_extents_impl<std::remove_cvref_t<T>> {};

template <typename T> constexpr bool is_extents_v = is_extents<T>::value;

template <typename T>
concept extents_c = is_extents_v<T>;

namespace detail {

template <typename T> struct is_extents_tuple_impl : std::false_type {};

template <typename... ExtentsTypes>
struct is_extents_tuple_impl<std::tuple<ExtentsTypes...>>
    : std::conjunction<is_extents<ExtentsTypes>...> {};

} // namespace detail

template <typename T>
struct is_extents_tuple
    : detail::is_extents_tuple_impl<std::remove_cvref_t<T>> {};

template <typename T>
constexpr bool is_extents_tuple_v = is_extents_tuple<T>::value;

template <typename T>
concept extents_tuple_c = is_extents_tuple_v<T>;

template <typename T>
concept extents_info_c = extents_c<T> || extents_tuple_c<T>;

namespace detail {

template <typename T> struct is_mdspan_impl : std::false_type {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename AccessorType>
struct is_mdspan_impl<
    stdex::mdspan<ElementType, ExtentsType, LayoutType, AccessorType>>
    : std::true_type {};

} // namespace detail

template <typename T>
struct is_mdspan : detail::is_mdspan_impl<std::remove_cvref_t<T>> {};

template <typename T> constexpr bool is_mdspan_v = is_mdspan<T>::value;

template <typename T>
concept mdspan_c = requires {
    typename std::remove_cvref_t<T>::value_type;
    typename std::remove_cvref_t<T>::element_type;
    typename std::remove_cvref_t<T>::extents_type;
    typename std::remove_cvref_t<T>::layout_type;
    typename std::remove_cvref_t<T>::accessor_type;
} && is_mdspan_v<T>;

namespace detail {

template <typename T> struct is_mdarray_impl : std::false_type {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename ContainerType>
struct is_mdarray_impl<
    stdex::mdarray<ElementType, ExtentsType, LayoutType, ContainerType>>
    : std::true_type {};

} // namespace detail

template <typename T>
struct is_mdarray : detail::is_mdarray_impl<std::remove_cvref_t<T>> {};

template <typename T> constexpr bool is_mdarray_v = is_mdarray<T>::value;

template <typename T>
concept mdarray_c = is_mdarray_v<T>;

template <typename T> constexpr bool is_md_v = mdspan_c<T> || mdarray_c<T>;

template <typename T>
concept md_c = is_md_v<T>;

template <size_t start, size_t end>
using slice = stdex::strided_slice<std::integral_constant<size_t, start>,
                                   std::integral_constant<size_t, end - start>,
                                   std::integral_constant<size_t, 1>>;

enum class MPMode : uint8_t {
    NONE,  // No parallelization
    SIMD,  // SIMD parallelization
    CPUMP, // CPU multi-processing with OpenMP
};

// ----------------------------------------------------------------------

namespace detail {

template <typename T, typename = void> struct value_type_t_impl {
    using type = std::remove_cvref_t<T>;
};

template <typename T>
struct value_type_t_impl<
    T, std::void_t<typename std::remove_cvref_t<T>::value_type>> {
    using type = typename std::remove_cvref_t<T>::value_type;
};

} // namespace detail

template <typename T>
using value_type_t = typename detail::value_type_t_impl<T>::type;

// ----------------------------------------------------------------------

namespace detail {

template <size_t Size> struct signed_by_size;

template <> struct signed_by_size<1> {
    using type = int8_t;
};

template <> struct signed_by_size<2> {
    using type = int16_t;
};

template <> struct signed_by_size<4> {
    using type = int32_t;
};

template <> struct signed_by_size<8> {
    using type = int64_t;
};

template <size_t Size>
using signed_by_size_t = typename signed_by_size<Size>::type;

template <typename T>
inline constexpr bool valid_extent_index_v =
    std::integral<std::remove_cvref_t<T>> &&
    !std::same_as<std::remove_cvref_t<T>, bool> &&
    (sizeof(std::remove_cvref_t<T>) == 1 ||
     sizeof(std::remove_cvref_t<T>) == 2 ||
     sizeof(std::remove_cvref_t<T>) == 4 ||
     sizeof(std::remove_cvref_t<T>) == 8);

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
    static constexpr size_t size =
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

// ----------------------------------------------------------------------

namespace detail {

template <typename T> struct unwrap_optional {
    using type = std::remove_cvref_t<T>;
};

template <typename T> struct unwrap_optional<std::optional<T>> {
    using type = std::remove_cvref_t<T>;
};

template <typename T>
using data_arg_t = typename unwrap_optional<std::remove_cvref_t<T>>::type;

template <typename... Ts> struct filter_nullopt;

template <> struct filter_nullopt<> {
    using type = std::tuple<>;
};

template <typename T, typename... Ts> struct filter_nullopt<T, Ts...> {
  private:
    using arg_t = data_arg_t<T>;
    using tail_t = typename filter_nullopt<Ts...>::type;

  public:
    using type = std::conditional_t<
        std::same_as<std::remove_cvref_t<T>, std::nullopt_t>, tail_t,
        decltype(std::tuple_cat(std::declval<std::tuple<arg_t>>(),
                                std::declval<tail_t>()))>;
};

template <typename T>
inline constexpr bool data_bool_v = std::same_as<std::remove_cvref_t<T>, bool>;

template <typename T>
inline constexpr bool data_fpoint_v =
    std::floating_point<std::remove_cvref_t<T>>;

template <typename T>
inline constexpr bool data_integer_v =
    std::integral<std::remove_cvref_t<T>> && !data_bool_v<T>;

template <typename T>
inline constexpr bool data_scalar_v =
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

// floating + something -> floating
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

// integer + integer -> common_index_type_t
template <typename T1, typename T2>
    requires(data_integer_v<T1> && data_integer_v<T2> &&
             requires {
                 typename common_index_type_t<std::remove_cvref_t<T1>,
                                              std::remove_cvref_t<T2>>;
             })
struct common_data_pair_impl<T1, T2> {
    using type =
        common_index_type_t<std::remove_cvref_t<T1>, std::remove_cvref_t<T2>>;
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
  public:
    using type = typename data_promote_impl<
        typename common_data_pair_impl<std::remove_cvref_t<T1>,
                                       std::remove_cvref_t<T2>>::type,
        Ts...>::type;
};

template <typename Tuple> struct common_data_type_impl {
    // no type
};

template <typename... Ts>
struct common_data_type_impl<std::tuple<Ts...>> : data_promote_impl<Ts...> {};

} // namespace detail

template <typename... Ts>
using common_data_type_t = typename detail::common_data_type_impl<
    typename detail::filter_nullopt<Ts...>::type>::type;

// ----------------------------------------------------------------------

} // namespace core
} // namespace mdtensor
