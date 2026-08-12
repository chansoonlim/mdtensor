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

namespace mdtensor {
namespace detail {

template <bool concatenate, std::size_t... Extents>
[[nodiscard]] consteval std::size_t concatenate_static_extent() noexcept {
    static_assert(sizeof...(Extents) > 0,
                  "At least one extent must be provided for concatenation.");

    if constexpr (concatenate) {
        if constexpr (((Extents == core::dyn) || ...)) {
            return core::dyn;

        } else {
            return (Extents + ...);
        }

    } else {
        if constexpr (((Extents == core::dyn) && ...)) {
            return core::dyn;

        } else {
            constexpr std::size_t cext =
                std::max({((Extents != core::dyn) ? Extents : 0)...});

            static_assert(((Extents == cext || Extents == core::dyn) && ...),
                          "Incompatible static extents for concatenation.");

            return cext;
        }
    }
}

template <bool concatenate, typename index_t,
          std::convertible_to<index_t>... exts_t>
[[nodiscard]] constexpr index_t concatenate_extent(exts_t &&...exts) {
    static_assert(sizeof...(exts) > 0,
                  "At least one extent must be provided for concatenation.");

    if constexpr (concatenate) {
        return (exts + ...);

    } else {
        const index_t cext = std::get<0>(std::forward_as_tuple(exts...));

        if (((cext != static_cast<index_t>(exts)) && ...)) {
            throw std::invalid_argument(
                "Incompatible extents for concatenation.");
        }

        return cext;
    }
}

template <std::int64_t axis, core::extents_c... ins_t>
[[nodiscard]] constexpr auto concatenate_extents(ins_t &&...ins) {
    static_assert(sizeof...(ins) > 0,
                  "At least one extents must be provided for concatenation.");

    using index_t = core::common_integral_type_t<
        typename std::remove_cvref_t<ins_t>::index_type...>;

    constexpr std::size_t rank = std::remove_cvref_t<
        std::tuple_element_t<0, std::tuple<ins_t...>>>::rank();

    static_assert(((ins.rank() == rank) && ...),
                  "All input extents must have the same rank.");

    if constexpr (rank == 0) {
        return core::extents<index_t>{};

    } else {
        constexpr std::size_t baxis =
            static_cast<std::size_t>(core::bounding_index(axis, rank - 1));

        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            const auto static_extent_at = [&]<std::size_t I>() {
                return concatenate_static_extent<
                    I == baxis,
                    std::remove_cvref_t<ins_t>::static_extent(I)...>();
            };

            const auto extent_at = [&]<std::size_t I>() {
                return concatenate_extent<I == baxis, index_t>(
                    ins.extent(I)...);
            };

            return core::extents<
                index_t, static_extent_at.template operator()<Is>()...> {
                extent_at.template operator()<Is>()...
            };
        }(std::make_index_sequence<rank>{});
    }
}

} // namespace detail

template <std::int64_t axis = 0, typename dtype = void>
[[nodiscard]] constexpr auto concatenate(auto &&...ins) {
    if constexpr ((!core::mdspan_c<decltype(ins)> || ...)) {
        return concatenate<axis>(
            core::to_const_mdspan(std::forward<decltype(ins)>(ins))...);

    } else {
        constexpr std::size_t rank = std::remove_cvref_t<
            std::tuple_element_t<0, std::tuple<decltype(ins)...>>>::rank();
        constexpr std::size_t baxis =
            static_cast<std::size_t>(core::bounding_index(axis, rank - 1));

        // generate out extents
        const auto out_extents =
            detail::concatenate_extents<axis>(ins.extents()...);

        // generate out
        using value_t = core::output_value_t<
            dtype, typename std::remove_cvref_t<decltype(ins)>::value_type...>;
        auto out = empty<value_t>(out_extents);

        // concatenate
        using index_t = typename decltype(out_extents)::index_type;
        index_t offset = 0;

        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&] {
                 const auto in = std::get<Is>(std::forward_as_tuple(
                     std::forward<decltype(ins)>(ins)...));

                 const index_t extent = static_cast<index_t>(in.extent(baxis));

                 static_cast<void>(copy(in, core::submdspan_from_left<baxis>(
                                                out, core::stdex::strided_slice{
                                                         offset, extent, 1})));

                 offset += extent;
             })(),
             ...);
        }(std::make_index_sequence<sizeof...(ins)>{});

        return out;
    }
}

} // namespace mdtensor
