/**
 * @file
 * @brief Type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// TODO: Remove when C++23 std::mdspan supports
#ifndef MDSPAN_SINGLE_HEADER_INCLUDE_GUARD_ // for godbolt test
#include <experimental/mdarray>
#include <experimental/mdspan>
#endif

namespace mdtensor {

#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
#define REAL_GCC __GNUC__ // probably
#endif

// TODO: modify under define
#if defined(_OPENMP) && defined(REAL_GCC)
#define MDTENSOR_USE_OPENMP
#endif

namespace core {

namespace stdex = std::experimental;

// ------------------------------------------------------------------
// - general type aliases -------------------------------------------
// ------------------------------------------------------------------

template <typename T>
concept integral_c = std::integral<std::remove_cvref_t<T>>;

template <typename T>
concept unsigned_integral_c = std::unsigned_integral<std::remove_cvref_t<T>>;

template <typename T>
concept floating_point_c = std::floating_point<std::remove_cvref_t<T>>;

template <typename T>
concept arithmetic = std::integral<std::remove_cvref_t<T>> ||
                     std::floating_point<std::remove_cvref_t<T>>;

template <typename T>
concept arithmetic_c = arithmetic<std::remove_cvref_t<T>>;

// ------------------------------------------------------------------
// - extents type aliases -------------------------------------------
// ------------------------------------------------------------------

template <typename IndexType, std::size_t... Extents>
using extents = stdex::extents<IndexType, Extents...>;

template <typename IndexType, std::size_t Rank>
using dextents = stdex::dextents<IndexType, Rank>;

#if false
// dims: will be included in C++23
// (https://en.cppreference.com/w/cpp/container/mdspan/extents)
template <std::size_t Rank, class IndexType = std::size_t>
using dims = stdex::dims<IndexType, Rank>;

#else
template <std::size_t Rank, class IndexType = std::size_t>
using dims = stdex::dextents<IndexType, Rank>;

#endif

template <typename T> struct is_extents : stdex::detail::__is_extents<T> {};

template <typename T> constexpr bool is_extents_v = is_extents<T>::value;

template <typename T>
concept extents_c = is_extents_v<std::remove_cvref_t<T>>;

namespace detail {

template <typename T> struct is_extents_tuple_impl : std::false_type {};

template <typename... Ts>
struct is_extents_tuple_impl<std::tuple<Ts...>>
    : std::conjunction<is_extents<std::remove_cvref_t<Ts>>...> {};

} // namespace detail

template <typename T>
struct is_extents_tuple : detail::is_extents_tuple_impl<T> {};

template <typename T>
constexpr bool is_extents_tuple_v = is_extents_tuple<T>::value;

template <typename T>
concept extents_tuple_c = is_extents_tuple_v<T>;

constexpr auto dynamic_extent = stdex::dynamic_extent;
constexpr auto dyn = dynamic_extent;

// ------------------------------------------------------------------
// - mdspan type aliases --------------------------------------------
// ------------------------------------------------------------------

template <typename ElementType, typename Extents,
          typename LayoutPolicy = stdex::layout_right,
          typename AccessorPolicy = stdex::default_accessor<ElementType>>
using mdspan =
    stdex::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

namespace detail {

template <typename T> struct is_mdspan_impl : std::false_type {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename AccessorType>
struct is_mdspan_impl<
    mdspan<ElementType, ExtentsType, LayoutType, AccessorType>>
    : std::true_type {};

} // namespace detail

template <typename T> struct is_mdspan : detail::is_mdspan_impl<T> {};

template <typename T> constexpr bool is_mdspan_v = is_mdspan<T>::value;

template <typename T>
concept mdspan_c = is_mdspan_v<std::remove_cvref_t<T>>;

constexpr auto full_extent = stdex::full_extent;

// ------------------------------------------------------------------
// - mdarray type aliases ------------------------------------------
// ------------------------------------------------------------------

template <typename ElementType, typename Extents,
          typename LayoutPolicy = stdex::layout_right,
          typename Container = std::vector<ElementType>>
using mdarray = stdex::mdarray<ElementType, Extents, LayoutPolicy, Container>;

namespace detail {

template <typename T> struct is_mdarray_impl : std::false_type {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename ContainerType>
struct is_mdarray_impl<
    mdarray<ElementType, ExtentsType, LayoutType, ContainerType>>
    : std::true_type {};

} // namespace detail

template <typename T> struct is_mdarray : detail::is_mdarray_impl<T> {};

template <typename T> constexpr bool is_mdarray_v = is_mdarray<T>::value;

template <typename T>
concept mdarray_c = is_mdarray_v<std::remove_cvref_t<T>>;

// ------------------------------------------------------------------
// - backend type aliases ---------------------------------------------
// ------------------------------------------------------------------

enum class Backend {
    AUTO,   // Automatically select backend based on input types and sizes
    NATIVE, // Native mdtensor implementation
    SIMD,   // SIMD parallelization

#ifdef MDTENSOR_USE_EIGEN
    EIGEN, // Eigen backend
#endif

#ifdef MDTENSOR_USE_OPENMP
    OPENMP, // CPU multi-processing with OpenMP
#endif
};

namespace detail {

template <typename T> struct is_backend_impl : std::false_type {};

template <> struct is_backend_impl<Backend> : std::true_type {};

} // namespace detail

template <typename T> struct is_backend : detail::is_backend_impl<T> {};

template <typename T> constexpr bool is_backend_v = is_backend<T>::value;

template <typename T>
concept backend_c = is_backend_v<std::remove_cvref_t<T>>;

// ------------------------------------------------------------------
// - copy type aliases ----------------------------------------------
// ------------------------------------------------------------------

enum class Copy : std::uint8_t {
    TRUE,  // Copy the input tensor to a new tensor
    FALSE, // Do not copy the input tensor; return a view of the input tensor
    AUTO,  // Automatically determine whether to copy or not based on input
};

// ------------------------------------------------------------------
// - strided_slice type aliases -------------------------------------
// ------------------------------------------------------------------

template <std::size_t start, std::size_t end>
using slice =
    stdex::strided_slice<std::integral_constant<std::size_t, start>,
                         std::integral_constant<std::size_t, end - start>,
                         std::integral_constant<std::size_t, 1>>;

// ------------------------------------------------------------------
// - is_nullopt_t type aliases --------------------------------------
// ------------------------------------------------------------------

namespace detail {

template <typename T> struct is_nullopt_impl : std::false_type {};

template <> struct is_nullopt_impl<std::nullopt_t> : std::true_type {};

} // namespace detail

template <typename T> struct is_nullopt_t : detail::is_nullopt_impl<T> {};

template <typename T> constexpr bool is_nullopt_t_v = is_nullopt_t<T>::value;

template <typename T>
concept is_nullopt_t_c = is_nullopt_t_v<std::remove_cvref_t<T>>;

// ------------------------------------------------------------------
// - common_index_type_t helpers ------------------------------------
// ------------------------------------------------------------------

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

// ------------------------------------------------------------------
// - common_data_type_t helpers -------------------------------------
// ------------------------------------------------------------------

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

} // namespace core
} // namespace mdtensor
