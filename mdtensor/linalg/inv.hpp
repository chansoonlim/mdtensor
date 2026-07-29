/**
 * @file
 * @brief Matrix inverse utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../creation/copy.hpp"
#include "../creation/eye.hpp"
#include "../math/absolute.hpp"

namespace mdtensor {
namespace linalg {
namespace detail {

template <core::md_c in_t, core::md_c out_t>
[[nodiscard]] inline constexpr bool inv_naive(in_t &&in, out_t &&out) {
    auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    static_assert(decltype(in_mds)::rank() == 2);
    static_assert(decltype(out_mds)::rank() == 2);

    using index_t = typename decltype(in_mds)::index_type;

    const index_t n = in_mds.extent(0);

    if (in_mds.extent(0) != in_mds.extent(1) ||
        in_mds.extent(0) != out_mds.extent(0) ||
        in_mds.extent(0) != out_mds.extent(1)) {
        return false;
    }

    auto in_copy = copy(in_mds);
    eye_to(out_mds);

    for (index_t i = 0; i < n; i++) {
        index_t pivot_row = i;
        auto max_abs = absolute(in_copy(i, i));

        for (index_t row = i + 1; row < n; row++) {
            const auto candidate = absolute(in_copy(row, i));

            if (candidate > max_abs) {
                max_abs = candidate;
                pivot_row = row;
            }
        }

        if (max_abs == 0) {
            // Handle error: singular matrix (no inverse)
            return false;
        }

        if (pivot_row != i) {
            for (index_t j = 0; j < n; j++) {
                std::swap(in_copy(i, j), in_copy(pivot_row, j));
                std::swap(out_mds(i, j), out_mds(pivot_row, j));
            }
        }

        const auto pivot = in_copy(i, i);

        for (index_t j = 0; j < n; j++) {
            in_copy(i, j) /= pivot;
            out_mds(i, j) /= pivot;
        }

        for (index_t j = 0; j < n; j++) {
            if (i == j) {
                continue;
            }

            const auto factor = in_copy(j, i);

            if (factor == 0) {
                continue;
            }

            for (index_t k = 0; k < n; k++) {
                in_copy(j, k) -= factor * in_copy(i, k);
                out_mds(j, k) -= factor * out_mds(i, k);
            }
        }
    }

    return true;
}

template <core::md_c in_t, core::md_c out_t>
[[nodiscard]] inline constexpr bool inv_impl(in_t &&in, out_t &&out) {
    static_assert(std::remove_cvref_t<in_t>::rank() == 2);
    static_assert(std::remove_cvref_t<out_t>::rank() == 2);

#ifdef MDTENSOR_USE_EIGEN
#if __cplusplus >= 202302L // TODO: Impliement for C++20
    if constexpr (core::eigen::eigen_mappable_mdspan_c<in_t> &&
                  core::eigen::eigen_mappable_mdspan_c<out_t>) {
        if (!std::is_constant_evaluated()) [[likely]] {
            const auto ein = core::eigen::to_eigen(in);
            auto eout = core::eigen::to_eigen(out);

            eout = ein.inverse();

            return true;
        }
    }

#else
    assert(false && "Eigen inverse not implemented for C++20");

#endif
#endif

    return inv_naive(in, out);
}

} // namespace detail

template <core::MPMode mpmode = core::MPMode::NONE, typename in_t,
          typename out_t, typename valid_t>
inline constexpr void inv_to(in_t &&in, out_t &&out, valid_t &&valid) {
    core::batch<mpmode>(
        [](auto &&in, auto &&out, auto &&valid) {
            valid() = detail::inv_impl(std::forward<decltype(in)>(in),
                                       std::forward<decltype(out)>(out));
        },
        std::index_sequence<2, 2, 0>{},
        std::integer_sequence<bool, false, true, true>{},
        std::forward<in_t>(in), std::forward<out_t>(out),
        std::forward<valid_t>(valid));
}

template <typename dtype = void, core::MPMode mpmode = core::MPMode::NONE,
          typename in_t>
[[nodiscard]] inline constexpr auto inv(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    auto out = core::create_out<dtype>(
        std::index_sequence<2>{},
        core::slice_extents_from_right<2>(in_mds.extents()), in_mds);
    auto valid = core::create_out<bool>(
        std::index_sequence<2>{}, core::stdex::extents<uint8_t>{}, in_mds);

    inv_to<mpmode>(in_mds, out, valid);

    return std::pair{out, valid};
}

} // namespace linalg
} // namespace mdtensor
