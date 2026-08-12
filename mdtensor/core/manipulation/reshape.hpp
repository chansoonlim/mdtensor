/**
 * @file
 * @brief Reshape utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../tensor/tensor.hpp"
#include "../ufunc/ufunc.hpp"

namespace mdtensor::core {

enum class Copy {
    TRUE,  // Copy the input tensor to a new tensor
    FALSE, // Do not copy the input tensor; return a view of the input tensor
    AUTO,  // Automatically determine whether to copy or not based on input
};

namespace detail {

template <typename in_t>
    requires(mdspan_c<in_t> || mdarray_c<in_t>)
[[nodiscard]] consteval bool is_always_c_contiguous() noexcept {
    return std::same_as<typename std::remove_cvref_t<in_t>::layout_type,
                        stdex::layout_right>;
}

template <mdspan_c in_t>
[[nodiscard]] constexpr bool is_c_contiguous(const in_t &in) noexcept {
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

template <extents_c exts_t>
[[nodiscard]] constexpr auto make_reshape_view(auto &&in, exts_t &&exts) {
    const auto in_mds = to_mdspan(std::forward<decltype(in)>(in));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using new_exts_t = std::remove_cvref_t<exts_t>;

    using out_mds_t =
        mdspan<typename in_mds_t::element_type, new_exts_t, stdex::layout_right,
               typename in_mds_t::accessor_type>;

    if constexpr (is_always_same_extents<typename in_mds_t::extents_type,
                                         new_exts_t>()) {
        return in_mds;

    } else {
        return out_mds_t{
            in_mds.data_handle(),
            typename out_mds_t::mapping_type{std::forward<exts_t>(exts)},
            in_mds.accessor()};
    }
}

template <extents_c exts_t>
[[nodiscard]] constexpr auto make_reshape_copy(auto &&in, exts_t &&exts) {
    const auto in_mds = to_const_mdspan(std::forward<decltype(in)>(in));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using value_t = typename in_mds_t::value_type;

    auto out = make_tensor<value_t>(std::forward<exts_t>(exts));

    batch([&](auto &&in, auto &&out) { out = in; }, in_mds,
          make_reshape_view(out, in_mds.extents()));

    return out;
}

} // namespace detail

template <Copy copy = Copy::AUTO>
[[nodiscard]] constexpr auto reshape(auto &&in, auto &&shape) {
    const auto in_mds = to_mdspan(std::forward<decltype(in)>(in));
    const auto exts = to_extents(std::forward<decltype(shape)>(shape));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using in_exts_t = typename in_mds_t::extents_type;
    using exts_t = std::remove_cvref_t<decltype(exts)>;

    if constexpr (in_exts_t::rank_dynamic() == 0 &&
                  exts_t::rank_dynamic() == 0) {
        static_assert(extents_size(in_exts_t{}) == extents_size(exts_t{}),
                      "Reshape error: input and output extents "
                      "must have the same size.");

    } else if (extents_size(in_mds.extents()) != extents_size(exts)) {
        throw std::invalid_argument("Reshape error: input and output "
                                    "extents must have the same size.");
    }

    constexpr bool can_borrow =
        std::is_lvalue_reference_v<decltype(in)> || mdspan_c<decltype(in)>;

    if constexpr (copy == Copy::AUTO) {
        if constexpr (can_borrow &&
                      detail::is_always_c_contiguous<in_mds_t>()) {
            return reshape<Copy::FALSE>(in_mds, exts);

        } else {
            return reshape<Copy::TRUE>(in_mds, exts);
        }

    } else if constexpr (copy == Copy::TRUE) {
        return detail::make_reshape_copy(in_mds, exts);

    } else if constexpr (copy == Copy::FALSE) {
        static_assert(can_borrow, "Reshape error: zero-copy reshape cannot "
                                  "bind to a temporary owning tensor.");

        if (!detail::is_c_contiguous(in_mds)) {
            throw std::invalid_argument("Reshape error: zero-copy reshape "
                                        "requires a C-contiguous input.");
        }

        if constexpr (is_always_same_extents<in_exts_t, exts_t>()) {
            return std::forward<decltype(in)>(in);

        } else {
            return detail::make_reshape_view(in_mds, exts);
        }
    }
}

} // namespace mdtensor::core
