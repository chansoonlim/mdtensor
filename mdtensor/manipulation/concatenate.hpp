/**
 * @file
 * @brief Concatenation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../creation/copy.hpp"
#include "../creation/empty.hpp"
#include "expand_dims.hpp"

namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, size_t Is, size_t axis>
constexpr size_t concatenate_static_extent() {
    constexpr size_t e1 = in1_t::static_extent(Is);
    constexpr size_t e2 = in2_t::static_extent(Is);

    if constexpr (Is == axis) {
        return (e1 != dyn && e2 != dyn) ? (e1 + e2) : dyn;

    } else {
        static_assert(e1 == e2 || e1 == dyn || e2 == dyn,
                      "Incompatible extents for concatenate.");
        return (e1 != dyn && e2 != dyn) ? e1 : dyn;
    }
}

template <int64_t Axis, extents_c in1_t, extents_c in2_t, extents_c... ins_t>
    requires(in1_t::rank() == in2_t::rank())
[[nodiscard]] inline constexpr auto
concatenate_extents(const in1_t &in1 = in1_t{}, const in2_t &in2 = in2_t{},
                    const ins_t &...ins) noexcept {
    constexpr size_t rank = in1_t::rank();
    constexpr size_t axis = static_cast<size_t>(
        ((Axis % static_cast<int64_t>(rank)) + (rank)) % rank);

    using index_t = core::extent_common_type_t<typename in1_t::index_type,
                                               typename in2_t::index_type>;

    const auto exts = [&]<typename E1, typename E2>(const E1 &e1,
                                                    const E2 &e2) {
        static_assert(E1::rank() == E2::rank());

        auto dyn_extent = [&]<std::size_t I>() -> index_t {
            const index_t d1 = static_cast<index_t>(e1.extent(I));
            const index_t d2 = static_cast<index_t>(e2.extent(I));

            if constexpr (I == axis) {
                return static_cast<index_t>(d1 + d2);

            } else {
                assert(d1 == d2);
                return d1;
            }
        };

        auto impl = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return extents<index_t,
                           concatenate_static_extent<E1, E2, Is, axis>()...>{
                dyn_extent.template operator()<Is>()...};
        };

        return impl(std::make_index_sequence<E1::rank()>{});
    }(in1, in2);

    if constexpr (sizeof...(ins_t) != 0) {
        return concatenate_extents<axis>(exts, ins...);

    } else {
        return exts;
    }
}

template <int64_t Axis, extents_c... ins_t>
[[nodiscard]] inline constexpr auto
concatenate_extents(std::tuple<ins_t...> &&ins) noexcept {
    constexpr size_t ins_num =
        std::tuple_size_v<std::remove_reference_t<decltype(ins)>>;

    static_assert(ins_num != 0, "concatenate requires at least one input.");

    if constexpr (ins_num == 1) {
        return std::get<0>(ins);

    } else {
        return std::apply(
            [&](auto &&...elems) {
                return concatenate_extents<Axis>(
                    std::forward<decltype(elems)>(elems)...);
            },
            ins);
    }
}

} // namespace detail

/**
 * @brief Concatenate multiple inputs along a specified axis (out-of-place).
 *
 * @tparam Axis Axis to concatenate. Negative values are supported and
 *         normalized by the input rank (NumPy-like semantics).
 * @tparam ins_t Input types (mdspan, mdarray, scalar, etc.).
 *
 * @param ins Input tensors to concatenate.
 *
 * @return Newly allocated mdarray containing the concatenation result.
 *
 * @note Rank-0 inputs (scalars) are expanded to rank-1 along the last axis
 *       using expand_dims<-1>.
 * @note All inputs must have the same rank after scalar expansion.
 * @note For dimensions other than Axis, extents must match.
 * @note The output dtype is the common_type of all input value types.
 */
template <int64_t Axis, typename... ins_t>
[[nodiscard]] inline constexpr auto concatenate(ins_t &&...ins) noexcept {
    constexpr size_t num_ins = sizeof...(ins_t);
    auto ins_tuple = std::forward_as_tuple(ins...);

    // generate input mdspans
    const auto ins_mds = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::make_tuple([&]() {
            auto mds = core::to_const_mdspan(std::get<Is>(ins_tuple));

            if constexpr (mds.rank() == 0) {
                // if the input is a scalar, expand it to a 1D mdspan.
                return expand_dims<-1>(mds);

            } else {
                return mds;
            }
        }()...);
    }(std::make_index_sequence<num_ins>{});

    using ins_mds_t = decltype(ins_mds);

    // generate out extents
    constexpr size_t rank = std::tuple_element_t<0, ins_mds_t>::rank();
    constexpr size_t axis = static_cast<size_t>(
        ((Axis % static_cast<int64_t>(rank)) + (rank)) % rank);

    const auto out_extents =
        [&ins_mds]<size_t... Is>(std::index_sequence<Is...>) {
            return detail::concatenate_extents<axis>(
                std::make_tuple(std::get<Is>(ins_mds).extents()...));
        }(std::make_index_sequence<num_ins>{});

    // generate out
    using dtype = decltype([]<size_t... Is>(std::index_sequence<Is...>) {
        return core::data_common_type_t<
            core::value_type_t<std::tuple_element_t<Is, ins_mds_t>>...>{};
    }(std::make_index_sequence<num_ins>{}));
    auto out = empty<dtype>(out_extents);

    // concatenate
    [&ins_mds, &out]<size_t... Is>(std::index_sequence<Is...>) {
        (([&] {
             const size_t offset =
                 [&]<size_t... Js>(std::index_sequence<Js...>) {
                     return (0 + ... + std::get<Js>(ins_mds).extent(axis));
                 }(std::make_index_sequence<Is>{});
             const size_t extent = std::get<Is>(ins_mds).extent(axis);
             constexpr size_t stride = 1;

             copy_to(std::get<Is>(ins_mds),
                     [&]<size_t... Js>(std::index_sequence<Js...>) {
                         return core::submdspan_from_left(
                             core::to_mdspan(out), ((void)Js, full_extent)...,
                             strided_slice{offset, extent, stride});
                     }(std::make_index_sequence<axis>{}));
         })(),
         ...);
    }(std::make_index_sequence<num_ins>{});

    return out;
}

} // namespace mdtensor
