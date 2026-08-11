/**
 * @file
 * @brief Mdspan utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "common.hpp"

namespace mdtensor::core {

template <typename ElementType, typename Extents,
          typename LayoutPolicy = stdex::layout_right,
          typename AccessorPolicy = stdex::default_accessor<ElementType>>
using mdspan =
    stdex::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

namespace detail {

template <typename T> struct is_mdspan_impl : std::false_type {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename AccessorType>
struct is_mdspan_impl<
    mdspan<ElementType, ExtentsType, LayoutType, AccessorType>>
    : std::true_type {};

} // namespace detail

template <typename T> struct is_mdspan : detail::is_mdspan_impl<T> {};

template <typename T> constexpr bool is_mdspan_v = is_mdspan<T>::value;

template <typename T>
concept mdspan_c = is_mdspan_v<std::remove_cvref_t<T>>;

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
        using extents_t = extents<std::uint8_t>;

        return mdspan<element_t, extents_t>{std::addressof(io), extents_t{}};
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

            return mdspan<
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

        return mdspan<
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

} // namespace mdtensor::core
