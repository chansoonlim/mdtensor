/**
 * @file
 * @brief Mdspan utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "type.hpp"

namespace mdtensor::core {

[[nodiscard]] constexpr auto to_mdspan(auto &&io) {
    if constexpr (mdspan_c<decltype(io)>) {
        // If the input is already an mdspan, just return it as-is
        return std::forward<decltype(io)>(io);

    } else if constexpr (requires { io.to_mdspan(); }) {
        // If the input has a to_mdspan() member function, call it
        return io.to_mdspan();

    } else {
        // If the input is not an mdspan,
        // create a new mdspan that points to the input data
        using element_t = std::remove_reference_t<decltype(io)>;
        using extents_t = core::extents<std::uint8_t>;

        return core::mdspan<element_t, extents_t>{std::addressof(io),
                                                  extents_t{}};
    }
}

[[nodiscard]] constexpr auto to_const_mdspan(auto &&in) {
    if constexpr (mdspan_c<decltype(in)>) {
        if constexpr (std::is_const_v<typename std::remove_reference_t<
                          decltype(in)>::element_type>) {
            // If the input is already a const mdspan, just return it as-is
            return std::forward<decltype(in)>(in);

        } else {
            // If the input is a non-const mdspan,
            // create a new const mdspan with the same data handle and mapping
            using in_t = std::remove_cvref_t<decltype(in)>;

            return core::mdspan<
                const typename in_t::value_type, typename in_t::extents_type,
                typename in_t::layout_type,
                stdex::default_accessor<const typename in_t::value_type>>(
                in.data_handle(), in.mapping(), {});
        }

    } else {
        // If the input is not an mdspan,
        // create a new const mdspan with the same data handle and mapping
        auto mds = to_mdspan(std::forward<decltype(in)>(in));

        using mds_t = std::remove_cvref_t<decltype(mds)>;

        return core::mdspan<
            const typename mds_t::value_type, typename mds_t::extents_type,
            typename mds_t::layout_type,
            stdex::default_accessor<const typename mds_t::value_type>>(
            mds.data_handle(), mds.mapping(), {});
    }
}

[[nodiscard]] constexpr auto to_output_mdspan(auto &&out) {
    if constexpr (mdspan_c<decltype(out)>) {
        static_assert(
            !std::is_const_v<
                typename std::remove_reference_t<decltype(out)>::element_type>,
            "Output mdspan must not be const");

        // If the output is already an mdspan, just return it as-is
        return std::forward<decltype(out)>(out);

    } else {
        static_assert(std::is_lvalue_reference_v<decltype(out)>,
                      "Output owners and scalars must be passed as lvalues.");

        return to_mdspan(std::forward<decltype(out)>(out));
    }
}

template <typename T>
using to_mdspan_t = decltype(to_mdspan(std::declval<T>()));

[[nodiscard]] constexpr auto submdspan(auto &&io, auto &&...slices) {
    return stdex::submdspan(to_mdspan(std::forward<decltype(io)>(io)),
                            std::forward<decltype(slices)>(slices)...);
}

template <std::size_t lspace = 0, std::size_t rspace = 0>
[[nodiscard]] constexpr auto submdspan_with_space(auto &&io, auto &&...slices) {
    return [&]<std::size_t... Is, std::size_t... Js>(
               std::index_sequence<Is...>, std::index_sequence<Js...>) {
        return submdspan(to_mdspan(std::forward<decltype(io)>(io)),
                         ((void)Is, core::full_extent)...,
                         std::forward<decltype(slices)>(slices)...,
                         ((void)Js, core::full_extent)...);
    }(std::make_index_sequence<lspace>{}, std::make_index_sequence<rspace>{});
}

template <std::size_t lspace = 0>
[[nodiscard]] constexpr auto submdspan_from_left(auto &&io, auto &&...slices) {
    using base_t = std::remove_reference_t<decltype(io)>;

    constexpr std::size_t rspace =
        to_mdspan_t<base_t>::rank() - (lspace + sizeof...(slices));

    return submdspan_with_space<lspace, rspace>(
        std::forward<decltype(io)>(io),
        std::forward<decltype(slices)>(slices)...);
}

template <std::size_t rspace = 0>
[[nodiscard]] constexpr auto submdspan_from_right(auto &&io, auto &&...slices) {
    using base_t = std::remove_reference_t<decltype(io)>;

    constexpr std::size_t lspace =
        to_mdspan_t<base_t>::rank() - (rspace + sizeof...(slices));

    return submdspan_with_space<lspace, rspace>(
        std::forward<decltype(io)>(io),
        std::forward<decltype(slices)>(slices)...);
}

} // namespace mdtensor::core
