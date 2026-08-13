/**
 * @file
 * @brief Broadcast utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../base/base.hpp"

namespace mdtensor::core {

[[nodiscard]] constexpr auto broadcast_to(auto &&in, auto &&shape) {
    const auto in_mds = to_const_mdspan(std::forward<decltype(in)>(in));
    const auto exts = to_extents(std::forward<decltype(shape)>(shape));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using exts_t = std::remove_cvref_t<decltype(exts)>;

    constexpr std::size_t org_rank = in_mds_t::rank();
    constexpr std::size_t new_rank = exts_t::rank();

    static_assert(org_rank <= new_rank, "Incompatible ranks for broadcasting.");

    if constexpr (is_always_same_extents<typename in_mds_t::extents_type,
                                         exts_t>()) {
        return in_mds; // change to const mdspan

    } else if constexpr (org_rank == 0) {
        using index_t = typename exts_t::index_type;

        auto new_strides = std::array<index_t, new_rank>{};

        for (std::size_t i = 0; i < new_rank; i++) {
            new_strides[i] = 0;
        }

        return mdspan<typename in_mds_t::element_type, exts_t,
                      stdex::layout_stride, typename in_mds_t::accessor_type>{
            in_mds.data_handle(),
            stdex::layout_stride::mapping{exts, new_strides}};

    } else {
        using index_t = typename exts_t::index_type;
        using cindex_t =
            std::common_type_t<typename in_mds_t::index_type, index_t>;

        // ni = new_rank - org_rank + oi
        const auto get_ni = [](std::size_t i) {
            return new_rank - org_rank + i;
        };

        // assertion
        static_assert(
            [&] {
                for (std::size_t i = 0; i < org_rank; i++) {
                    const auto src = in_mds_t::static_extent(i);
                    const auto dst = exts_t::static_extent(get_ni(i));

                    if (src != dyn && dst != dyn && src != dst && src != 1) {
                        return false;
                    }
                }
                return true;
            }(),
            "Incompatible extents for broadcasting.");

        for (std::size_t i = 0; i < org_rank; i++) {
            if (static_cast<cindex_t>(in_mds.extent(i)) !=
                    static_cast<cindex_t>(exts.extent(get_ni(i))) &&
                static_cast<cindex_t>(in_mds.extent(i)) != cindex_t{1}) {
                throw std::invalid_argument(
                    "Incompatible extents for broadcasting.");
            }
        }

        // calculation
        auto new_strides = std::array<index_t, new_rank>{};

        for (std::size_t i = 0; i < new_rank - org_rank; i++) {
            new_strides[i] = 0;
        }

        for (std::size_t i = 0; i < org_rank; i++) {
            if (static_cast<cindex_t>(in_mds.extent(i)) ==
                static_cast<cindex_t>(exts.extent(get_ni(i)))) {
                new_strides[get_ni(i)] = static_cast<index_t>(in_mds.stride(i));

            } else {
                new_strides[get_ni(i)] = 0;
            }
        }

        return mdspan<typename in_mds_t::element_type, exts_t,
                      stdex::layout_stride, typename in_mds_t::accessor_type>{
            in_mds.data_handle(),
            stdex::layout_stride::mapping{exts, new_strides}};
    }
}

} // namespace mdtensor::core
