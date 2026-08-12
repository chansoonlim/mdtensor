/**
 * @file
 * @brief Kokkos type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

// TODO: Remove when C++23 std::mdspan supports
#ifndef MDSPAN_SINGLE_HEADER_INCLUDE_GUARD_ // for godbolt test
#include <experimental/mdarray>
#include <experimental/mdspan>
#endif

namespace mdtensor {

// TODO: move to other header
#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
#define REAL_GCC __GNUC__ // probably
#endif

// TODO: modify under define
#if defined(_OPENMP) && defined(REAL_GCC)
#define MDTENSOR_USE_OPENMP
#endif

namespace core {

namespace stdex = std::experimental;

constexpr auto dynamic_extent = stdex::dynamic_extent;
constexpr auto dyn = dynamic_extent;
constexpr auto full_extent = stdex::full_extent;

template <typename IndexType, std::size_t... Extents>
using extents = stdex::extents<IndexType, Extents...>;

template <typename IndexType, std::size_t Rank>
using dextents = stdex::dextents<IndexType, Rank>;

// NOTE: dims will be included in C++23
// (https://en.cppreference.com/w/cpp/container/mdspan/extents)
template <std::size_t Rank, class IndexType = std::size_t>
using dims = dextents<IndexType, Rank>;

template <typename ElementType, typename Extents,
          typename LayoutPolicy = stdex::layout_right,
          typename AccessorPolicy = stdex::default_accessor<ElementType>>
using mdspan =
    stdex::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

template <typename ElementType, typename Extents,
          typename LayoutPolicy = stdex::layout_right,
          typename Container = std::vector<ElementType>>
using mdarray = stdex::mdarray<ElementType, Extents, LayoutPolicy, Container>;

} // namespace core
} // namespace mdtensor
