/**
 * @file
 * @brief Core utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "base/base.hpp"
#include "broadcast/broadcast.hpp"
#include "manipulation/manipulation.hpp"
#include "tensor/tensor.hpp"
#include "ufunc/ufunc.hpp"

namespace mdtensor {

constexpr auto dyn = core::dyn;

template <typename T>
concept extents_c = core::extents_c<T>;

template <typename T>
concept mdspan_c = core::mdspan_c<T>;

template <typename T>
concept mdarray_c = core::mdarray_c<T>;

template <std::size_t start, std::size_t end>
using slice = core::slice<start, end>;

template <typename IndexType, std::size_t... Extents>
using extents = core::extents<IndexType, Extents...>;

template <typename IndexType, std::size_t Rank>
using dextents = core::dextents<IndexType, Rank>;

template <std::size_t Rank, class IndexType = std::size_t>
using dims = core::dims<Rank, IndexType>;

template <typename ElementType, typename Extents,
          typename LayoutPolicy = core::stdex::layout_right,
          typename AccessorPolicy = core::stdex::default_accessor<ElementType>>
using mdspan = core::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

constexpr auto full_extent = core::full_extent;

template <typename value_t, extents_c extent_t>
using tensor = core::tensor<value_t, extent_t>;

using Backend = core::Backend;

template <extents_c... ins_t>
[[nodiscard]] constexpr bool is_always_same_extents() noexcept {
    return core::is_always_same_extents<ins_t...>();
}

template <extents_c... ins_t>
[[nodiscard]] constexpr bool is_same_extents(ins_t &&...ins) noexcept {
    return core::is_same_extents(std::forward<ins_t>(ins)...);
}

[[nodiscard]] constexpr auto to_mdspan(auto &&io) {
    return core::to_mdspan(std::forward<decltype(io)>(io));
}

[[nodiscard]] constexpr auto to_const_mdspan(auto &&in) {
    return core::to_const_mdspan(std::forward<decltype(in)>(in));
}

[[nodiscard]] constexpr auto to_output_mdspan(auto &&out) {
    return core::to_output_mdspan(std::forward<decltype(out)>(out));
}

[[nodiscard]] constexpr auto submdspan(auto &&io, auto &&...slices) {
    return core::submdspan(std::forward<decltype(io)>(io),
                           std::forward<decltype(slices)>(slices)...);
}

template <std::size_t lspace = 0, std::size_t rspace = 0>
[[nodiscard]] constexpr auto submdspan_with_space(auto &&io, auto &&...slices) {
    return core::submdspan_with_space<lspace, rspace>(
        std::forward<decltype(io)>(io),
        std::forward<decltype(slices)>(slices)...);
}

template <std::size_t lspace = 0>
[[nodiscard]] constexpr auto submdspan_from_left(auto &&io, auto &&...slices) {
    return core::submdspan_from_left<lspace>(
        std::forward<decltype(io)>(io),
        std::forward<decltype(slices)>(slices)...);
}

template <std::size_t rspace = 0>
[[nodiscard]] constexpr auto submdspan_from_right(auto &&io, auto &&...slices) {
    return core::submdspan_from_right<rspace>(
        std::forward<decltype(io)>(io),
        std::forward<decltype(slices)>(slices)...);
}

} // namespace mdtensor
