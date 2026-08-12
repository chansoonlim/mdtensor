/**
 * @file
 * @brief Broadcast extents utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../base/base.hpp"

namespace mdtensor::core {
namespace detail {

template <std::size_t I, std::size_t brank, extents_c in_t>
[[nodiscard]] consteval std::size_t aligned_static_extent() noexcept {
    using base_t = std::remove_cvref_t<in_t>;

    constexpr std::size_t rank = base_t::rank();

    static_assert(I < brank, "Index I must be less than broadcast rank brank.");
    static_assert(rank <= brank,
                  "Input rank must be less than or equal to broadcast rank.");

    if constexpr (I < brank - rank) {
        return 1;

    } else {
        return base_t::static_extent(I - (brank - rank));
    }
}

template <std::size_t... Extents>
[[nodiscard]] consteval std::size_t broadcast_static_extent() noexcept {
    static_assert(sizeof...(Extents) > 0,
                  "At least one extent must be provided for broadcasting.");

    if constexpr (((Extents == 1 || Extents == dyn) && ...)) {
        // return dyn if any extent is dyn, else return 1
        return std::max({Extents...});

    } else {
        // select the extent that is not 1 or dyn
        constexpr std::size_t bext =
            std::max({((Extents != 1 && Extents != dyn) ? Extents : 0)...});

        static_assert(
            ((Extents == bext || Extents == 1 || Extents == dyn) && ...),
            "Incompatible static extents for broadcasting.");

        return bext;
    }
}

template <std::size_t I, std::size_t brank, extents_c in_t>
[[nodiscard]] constexpr auto aligned_extent(in_t &&in) noexcept {
    using index_t = typename std::remove_cvref_t<in_t>::index_type;

    constexpr std::size_t rank = in.rank();

    static_assert(I < brank, "Index I must be less than broadcast rank brank.");
    static_assert(rank <= brank,
                  "Input rank must be less than or equal to broadcast rank.");

    if constexpr (I < brank - rank) {
        return index_t{1};

    } else {
        return in.extent(I - (brank - rank));
    }
}

template <typename index_t, std::convertible_to<index_t>... exts_t>
[[nodiscard]] constexpr index_t broadcast_extent(exts_t &&...exts) {
    static_assert(sizeof...(exts) > 0,
                  "At least one extent must be provided for broadcasting.");

    index_t bext = 1;

    for (const index_t &ext : {static_cast<index_t>(exts)...}) {
        if (ext == 1) {
            continue;

        } else if (bext == 1) {
            bext = ext;

        } else if (ext != bext) {
            throw std::invalid_argument(
                "Incompatible extents for broadcasting.");
        }
    }

    return bext;
}

} // namespace detail

template <extents_c... ins_t>
[[nodiscard]] constexpr auto broadcast_extents(ins_t &&...ins) {
    static_assert(sizeof...(ins) > 0,
                  "At least one extents must be provided for broadcasting.");

    using index_t = core::common_integral_type_t<
        typename std::remove_cvref_t<ins_t>::index_type...>;

    constexpr std::size_t brank = std::max({ins.rank()...});

    if constexpr (brank == 0) {
        return extents<index_t>{};

    } else {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            const auto static_extent_at = [&]<std::size_t I>() {
                return detail::broadcast_static_extent<
                    detail::aligned_static_extent<I, brank, ins_t>()...>();
            };

            const auto extent_at = [&]<std::size_t I>() {
                return detail::broadcast_extent<index_t>(
                    detail::aligned_extent<I, brank>(
                        std::forward<ins_t>(ins))...);
            };

            return extents<index_t,
                           static_extent_at.template operator()<Is>()...> {
                extent_at.template operator()<Is>()...
            };
        }(std::make_index_sequence<brank>{});
    }
}

} // namespace mdtensor::core
