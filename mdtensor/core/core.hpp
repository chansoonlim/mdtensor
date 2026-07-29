/**
 * @file
 * @brief Core utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#ifdef MDTENSOR_USE_EIGEN
#include "eigen/eigen.hpp"
#endif

#include "batch.hpp"
#include "broadcast.hpp"
#include "container.hpp"
#include "extents.hpp"
#include "mdspan.hpp"
#include "submdspan.hpp"
#include "type.hpp"

#include <cmath>

namespace mdtensor {

constexpr auto dyn = core::stdex::dynamic_extent;

template <typename T>
concept extents_c = core::extents_c<T>;

template <typename T>
concept mdspan_c = core::mdspan_c<T>;

template <typename T>
concept mdarray_c = core::mdarray_c<T>;

template <typename T>
concept md_c = core::md_c<T>;

template <size_t start, size_t end> using slice = core::slice<start, end>;

template <typename IndexType, size_t... Extents>
using extents = core::stdex::extents<IndexType, Extents...>;

template <typename IndexType, size_t Rank>
using dextents = core::stdex::dextents<IndexType, Rank>;

// dims: will be included in C++23
// (https://en.cppreference.com/w/cpp/container/mdspan/extents)
template <size_t Rank, class IndexType = size_t>
using dims = core::stdex::dextents<IndexType, Rank>;

template <typename ElementType, typename Extents,
          typename LayoutPolicy = core::stdex::layout_right,
          typename AccessorPolicy = core::stdex::default_accessor<ElementType>>
using mdspan =
    core::stdex::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

template <typename value_t, extents_c extent_t>
using container = core::container<value_t, extent_t>;

using MPMode = core::MPMode;

} // namespace mdtensor
