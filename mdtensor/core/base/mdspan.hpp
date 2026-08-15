/**
 * @file
 * @brief Mdspan utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "type/type.hpp"

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

template <mdspan_c io_t>
[[nodiscard]] constexpr decltype(auto) unwrap_scalar(io_t &&io) {
    using base_t = std::remove_cvref_t<io_t>;

    if constexpr (base_t::rank() == 0) {
        return std::forward<io_t>(io)();

    } else {
        return std::forward<io_t>(io);
    }
}

template <mdspan_c in_t>
[[nodiscard]] consteval bool is_always_c_contiguous() noexcept {
    return std::same_as<typename std::remove_cvref_t<in_t>::layout_type,
                        stdex::layout_right>;
}

template <mdspan_c in_t>
[[nodiscard]] constexpr bool is_c_contiguous(in_t &&in) noexcept {
    if constexpr (in.rank() == 0) {
        return true;

    } else {
        // Empty tensors have no observable element ordering.
        if (extents_size(in.extents()) == 0) {
            return true;
        }

        if (!in.is_unique() || !in.is_exhaustive() || !in.is_strided()) {
            return false;
        }

        std::size_t expected_stride = 1;

        for (std::size_t i = in.rank(); i-- > 0;) {
            const std::size_t extent = static_cast<std::size_t>(in.extent(i));

            if (extent > 1 &&
                static_cast<std::size_t>(in.stride(i)) != expected_stride) {
                return false;
            }

            expected_stride *= extent;
        }

        return true;
    }
}

} // namespace mdtensor::core
