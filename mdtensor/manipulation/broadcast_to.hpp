/**
 * @file
 * @brief Broadcast_to utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {

template <typename in_t, extents_c new_extents_t>
[[nodiscard]] inline constexpr auto
broadcast_to(in_t &&in,
             new_extents_t &&new_extents = new_extents_t{}) noexcept {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    using in_mds_base_t = std::remove_cvref_t<decltype(in_mds)>;
    using new_extents_base_t = std::remove_cvref_t<new_extents_t>;

    constexpr size_t org_rank = in_mds_base_t::rank();
    constexpr size_t new_rank = new_extents_base_t::rank();

    static_assert(org_rank <= new_rank, "Incompatible ranks for broadcasting.");

    if constexpr (core::same<typename in_mds_base_t::extents_type,
                             new_extents_base_t>()) {
        return in_mds;

    } else if constexpr (org_rank == 0) {
        using index_t = typename new_extents_base_t::index_type;

        auto new_strides = std::array<index_t, new_rank>{};

        for (size_t i = 0; i < new_rank; i++) {
            new_strides[i] = 0;
        }

        return mdspan<typename in_mds_base_t::element_type, new_extents_base_t,
                      layout_stride, typename in_mds_base_t::accessor_type>{
            in_mds.data_handle(),
            layout_stride::mapping{std::forward<new_extents_t>(new_extents),
                                   new_strides}};

    } else {
        using index_t = typename new_extents_base_t::index_type;

        // ni = new_rank - org_rank + oi
        const auto get_ni = [](size_t i) { return new_rank - org_rank + i; };

        // assertion
        static_assert(
            [&] {
                for (size_t i = 0; i < org_rank; i++) {
                    if (in_mds_base_t::static_extent(i) !=
                            new_extents_base_t::static_extent(get_ni(i)) &&
                        in_mds_base_t::static_extent(i) != 1 &&
                        new_extents_base_t::static_extent(get_ni(i)) != dyn) {
                        return false;
                    }
                }
                return true;
            }(),
            "Incompatible extents for broadcasting.");

        for (size_t i = 0; i < org_rank; i++) {
            assert(static_cast<size_t>(in_mds.extent(i)) ==
                       static_cast<size_t>(new_extents.extent(get_ni(i))) ||
                   static_cast<size_t>(in_mds.extent(i)) == 1);
        }

        // calculation
        auto new_strides = std::array<index_t, new_rank>{};

        for (size_t i = 0; i < new_rank - org_rank; i++) {
            new_strides[i] = 0;
        }

        for (size_t i = 0; i < org_rank; i++) {
            new_strides[get_ni(i)] = static_cast<index_t>(in_mds.stride(i));
        }

        return mdspan<typename in_mds_base_t::element_type, new_extents_base_t,
                      layout_stride, typename in_mds_base_t::accessor_type>{
            in_mds.data_handle(),
            layout_stride::mapping{std::forward<new_extents_t>(new_extents),
                                   new_strides}};
    }
}

} // namespace mdtensor
