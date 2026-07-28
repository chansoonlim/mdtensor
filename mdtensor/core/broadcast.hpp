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
#include "type.hpp"

namespace mdtensor {
namespace core {
namespace detail {

template <size_t I, size_t brank, extents_c in_t>
[[nodiscard]] inline constexpr size_t aligned_static_extent() noexcept {
    using base_t = std::remove_cvref_t<in_t>;

    constexpr size_t rank = base_t::rank();

    if constexpr (I < brank - rank) {
        return 1;

    } else {
        return base_t::static_extent(I - (brank - rank));
    }
}

template <size_t... Exts>
[[nodiscard]] inline constexpr size_t broadcast_static_extent() noexcept {
    static_assert(
        [&] {
            constexpr auto exts_arr = std::array{Exts...};

            for (size_t i = 0; i < exts_arr.size(); i++) {
                for (size_t j = i + 1; j < exts_arr.size(); j++) {
                    const size_t ei = exts_arr[i];
                    const size_t ej = exts_arr[j];

                    if (ei != ej &&                               //
                        ei != 1 && ei != stdex::dynamic_extent && //
                        ej != 1 && ej != stdex::dynamic_extent) {
                        return false;
                    }
                }
            }
            return true;
        }(),
        "Incompatible static extents for broadcasting.");

    return std::max({Exts...});
}

template <size_t I, size_t brank, extents_c in_t>
[[nodiscard]] inline constexpr auto aligned_extent(in_t &&in) noexcept {
    using base_t = std::remove_cvref_t<in_t>;

    constexpr size_t rank = base_t::rank();

    if constexpr (I < brank - rank) {
        return typename base_t::index_type{1};

    } else {
        return in.extent(I - (brank - rank));
    }
}

template <typename index_t, typename... exts_t>
[[nodiscard]] inline constexpr index_t
broadcast_extent(exts_t... exts) noexcept {
    assert([&] {
        const auto exts_arr = std::array{static_cast<index_t>(exts)...};

        for (size_t i = 0; i < exts_arr.size(); i++) {
            for (size_t j = i + 1; j < exts_arr.size(); j++) {
                const index_t ei = exts_arr[i];
                const index_t ej = exts_arr[j];

                if (ei != ej && ei != 1 && ej != 1) {
                    return false;
                }
            }
        }
        return true;
    }());

    return std::max({static_cast<index_t>(exts)...});
}

} // namespace detail

template <extents_c... ins_t>
[[nodiscard]] inline constexpr auto broadcast_extents(ins_t &&...ins) noexcept {
    using index_t =
        common_index_type_t<typename std::remove_cvref_t<ins_t>::index_type...>;

    constexpr size_t brank = std::max({std::remove_cvref_t<ins_t>::rank()...});

    if constexpr (brank == 0) {
        return stdex::extents<index_t>{};

    } else {
        return [&]<size_t... Is>(std::index_sequence<Is...>) {
            const auto static_extent_at = [&]<size_t I>() {
                return detail::broadcast_static_extent<
                    detail::aligned_static_extent<I, brank, ins_t>()...>();
            };

            const auto extent_at = [&]<size_t I>() {
                return detail::broadcast_extent<index_t>(
                    detail::aligned_extent<I, brank>(
                        std::forward<ins_t>(ins))...);
            };

            return stdex::extents<
                index_t, static_extent_at.template operator()<Is>()...> {
                extent_at.template operator()<Is>()...
            };
        }(std::make_index_sequence<brank>{});
    }
}

template <typename in_t, extents_c new_extents_t>
[[nodiscard]] inline constexpr auto
broadcast_to(in_t &&in, new_extents_t &&new_extents) noexcept {
    const auto in_mds = to_const_mdspan(std::forward<in_t>(in));

    using in_mds_base_t = std::remove_cvref_t<decltype(in_mds)>;
    using new_extents_base_t = std::remove_cvref_t<new_extents_t>;

    constexpr size_t org_rank = in_mds_base_t::rank();
    constexpr size_t new_rank = new_extents_base_t::rank();

    static_assert(org_rank <= new_rank, "Incompatible ranks for broadcasting.");

    if constexpr (in_mds_base_t::rank_dynamic() == 0 &&
                  new_extents_base_t::rank_dynamic() == 0 &&
                  same_extents(typename in_mds_base_t::extents_type{},
                               new_extents_base_t{})) {
        return in_mds;

    } else if constexpr (org_rank == 0) {
        using index_t = typename new_extents_base_t::index_type;

        auto new_strides = std::array<index_t, new_rank>{};

        for (size_t i = 0; i < new_rank; i++) {
            new_strides[i] = 0;
        }

        return stdex::mdspan<typename in_mds_base_t::element_type,
                             new_extents_base_t, stdex::layout_stride,
                             typename in_mds_base_t::accessor_type>{
            in_mds.data_handle(),
            stdex::layout_stride::mapping{
                std::forward<new_extents_t>(new_extents), new_strides}};

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
                        new_extents_base_t::static_extent(get_ni(i)) !=
                            stdex::dynamic_extent) {
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
            if (static_cast<size_t>(in_mds.extent(i)) ==
                static_cast<size_t>(new_extents.extent(get_ni(i)))) {
                new_strides[get_ni(i)] = static_cast<index_t>(in_mds.stride(i));

            } else {
                new_strides[get_ni(i)] = 0;
            }
        }

        return stdex::mdspan<typename in_mds_base_t::element_type,
                             new_extents_base_t, stdex::layout_stride,
                             typename in_mds_base_t::accessor_type>{
            in_mds.data_handle(),
            stdex::layout_stride::mapping{
                std::forward<new_extents_t>(new_extents), new_strides}};
    }
}

namespace detail {

template <size_t... uranks, typename ios_mds_t>
[[nodiscard]] inline constexpr auto
get_broadcast_extents(std::index_sequence<uranks...>,
                      ios_mds_t &&ios_mds) noexcept {
    constexpr size_t ios_num =
        std::tuple_size_v<std::remove_cvref_t<ios_mds_t>>;

    // calculate ranks
    constexpr auto fr = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array{
            std::tuple_element_t<Is,
                                 std::remove_cvref_t<ios_mds_t>>::rank()...};
    }(std::make_index_sequence<ios_num>{});

    constexpr auto ur = std::array{uranks...};

    constexpr auto br = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array{(fr[Is] - ur[Is])...};
    }(std::make_index_sequence<ios_num>{});

    // calculate broadcasted extents
    const auto bexts = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return broadcast_extents(slice_extents_from_left<br[Is]>(
            std::get<Is>(ios_mds).extents())...);
    }(std::make_index_sequence<ios_num>{});

    return bexts;
}

} // namespace detail

template <size_t... uranks, bool... writable, typename... ios_t>
[[nodiscard]] inline constexpr auto
broadcast(std::index_sequence<uranks...>,
          std::integer_sequence<bool, writable...>, ios_t &&...ios) noexcept {
    static_assert(sizeof...(uranks) == sizeof...(ios_t));
    static_assert(sizeof...(writable) == sizeof...(ios_t));

    // calculate mdspans for inputs and outputs
    constexpr size_t ios_num = sizeof...(ios_t);
    const auto ios_mds =
        std::make_tuple(to_mdspan(std::forward<ios_t>(ios))...);

    // calculate broadcasted extents
    constexpr auto ur = std::array{uranks...};
    constexpr auto wa = std::array{writable...};
    const auto bexts = detail::get_broadcast_extents(
        std::index_sequence<uranks...>{}, ios_mds);

    // calculate broadcasted mdspans
    auto get_broadcasted = [&]<size_t I>() {
        if constexpr (wa[I]) {
            // writable elements does not apply broadcasting.
            return std::get<I>(ios_mds);

        } else {
            // read-only elements apply broadcasting.
            return broadcast_to(
                std::get<I>(ios_mds),
                compose_extents(bexts, slice_extents_from_right<ur[I]>(
                                           std::get<I>(ios_mds).extents())));
        }
    };

    return std::make_tuple(
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            return std::make_tuple(
                get_broadcasted.template operator()<Is>()...);
        }(std::make_index_sequence<ios_num>{}),
        bexts);
}

} // namespace core
} // namespace mdtensor
