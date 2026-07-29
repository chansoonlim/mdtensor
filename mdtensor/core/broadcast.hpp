/**
 * @file
 * @brief Broadcast utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "extents.hpp"
#include "mdspan.hpp"

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

    if constexpr (((Extents == 1 || Extents == core::dyn) && ...)) {
        // return dyn if any extent is dyn, else return 1
        return std::max({Extents...});

    } else {
        // select the extent that is not 1 or dyn
        constexpr std::size_t bext = std::max(
            {((Extents != 1 && Extents != core::dyn) ? Extents : 0)...});

        static_assert(
            ((Extents == bext || Extents == 1 || Extents == core::dyn) && ...),
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

    using index_t =
        common_index_type_t<typename std::remove_cvref_t<ins_t>::index_type...>;

    constexpr std::size_t brank = std::max({ins.rank()...});

    if constexpr (brank == 0) {
        return core::extents<index_t>{};

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

            return core::extents<
                index_t, static_extent_at.template operator()<Is>()...> {
                extent_at.template operator()<Is>()...
            };
        }(std::make_index_sequence<brank>{});
    }
}

[[nodiscard]] constexpr auto broadcast_to(auto &&in, auto &&shape) {
    const auto in_mds = to_const_mdspan(std::forward<decltype(in)>(in));
    const auto exts = core::to_extents(std::forward<decltype(shape)>(shape));

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

        return core::mdspan<typename in_mds_t::element_type, exts_t,
                            stdex::layout_stride,
                            typename in_mds_t::accessor_type>{
            in_mds.data_handle(),
            stdex::layout_stride::mapping{exts, new_strides}};

    } else {
        using index_t = typename exts_t::index_type;
        using cindex_t =
            common_index_type_t<typename in_mds_t::index_type, index_t>;

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

                    if (src != core::dyn && //
                        dst != core::dyn && //
                        src != dst &&       //
                        src != 1) {
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

        return core::mdspan<typename in_mds_t::element_type, exts_t,
                            stdex::layout_stride,
                            typename in_mds_t::accessor_type>{
            in_mds.data_handle(),
            stdex::layout_stride::mapping{exts, new_strides}};
    }
}

namespace detail {

template <std::size_t... uranks, core::mdspan_c... ios_t>
[[nodiscard]] constexpr auto
get_broadcast_extents(std::index_sequence<uranks...>, ios_t &&...ios) {
    static_assert(sizeof...(uranks) == sizeof...(ios_t),
                  "Number of uranks must match number of inputs.");
    static_assert(((ios.rank() >= uranks) && ...),
                  "Input rank must be greater than or equal to urank.");

    return broadcast_extents(
        slice_extents_from_left<ios.rank() - uranks>(ios.extents())...);
}

} // namespace detail

template <std::size_t... uranks, bool... bcast>
[[nodiscard]] constexpr auto broadcast(std::index_sequence<uranks...>,
                                       std::integer_sequence<bool, bcast...>,
                                       auto &&...ios) {
    static_assert(sizeof...(uranks) == sizeof...(ios));
    static_assert(sizeof...(bcast) == sizeof...(ios));

    // calculate mdspans for inputs and outputs
    const auto ios_mds =
        std::make_tuple(to_mdspan(std::forward<decltype(ios)>(ios))...);

    // calculate broadcasted extents
    constexpr auto ur = std::array{uranks...};

    const auto bexts = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return detail::get_broadcast_extents(std::index_sequence<ur[Is]...>{},
                                             std::get<Is>(ios_mds)...);
    }(std::make_index_sequence<sizeof...(ios)>{});

    // calculate broadcasted mdspans
    constexpr auto bc = std::array{bcast...};

    return std::make_tuple(
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            const auto get_broadcasted = [&]<std::size_t I>() {
                if constexpr (!bc[I]) {
                    // change to mdspan without broadcasting
                    return std::get<I>(ios_mds);

                } else if constexpr (bexts.rank() == 0) {
                    // change to const mdspan without broadcasting
                    return core::to_const_mdspan(std::get<I>(ios_mds));

                } else {
                    // broadcast to const mdspan
                    return broadcast_to(
                        std::get<I>(ios_mds),
                        compose_extents(bexts,
                                        slice_extents_from_right<ur[I]>(
                                            std::get<I>(ios_mds).extents())));
                }
            };

            return std::make_tuple(
                get_broadcasted.template operator()<Is>()...);
        }(std::make_index_sequence<sizeof...(ios)>{}),
        bexts);
}

} // namespace mdtensor::core
