/**
 * @file
 * @brief Linear system solve utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "lu.hpp"

namespace mdtensor {
namespace linalg {
namespace detail {

template <md_c a_t, md_c b_t, md_c x_t>
[[nodiscard]] inline constexpr bool solve_impl(a_t &&a, b_t &&b,
                                               x_t &&x) noexcept {
    auto a_mds = core::to_const_mdspan(std::forward<a_t>(a));
    auto b_mds = core::to_const_mdspan(std::forward<b_t>(b));
    auto x_mds = core::to_mdspan(std::forward<x_t>(x));

    using a_mds_t = std::remove_cvref_t<decltype(a_mds)>;
    using b_mds_t = std::remove_cvref_t<decltype(b_mds)>;
    using x_mds_t = std::remove_cvref_t<decltype(x_mds)>;

    static_assert(a_mds_t::rank() == 2);
    static_assert(b_mds_t::rank() == 1 || b_mds_t::rank() == 2);
    static_assert(x_mds_t::rank() == b_mds_t::rank());

    using index_t = typename a_mds_t::index_type;

    const index_t n = a_mds.extent(0);

    assert(a_mds.extent(1) == n);
    assert(b_mds.extent(0) == n);
    assert(x_mds.extent(0) == n);

    // LU decomposition of A
    const auto [p_indices, l, u] = lu_p_indices(a_mds);

    // check singularity
    for (index_t idx = 0; idx < n; idx++) {
        if (u(idx, idx) == 0) {
            return false;
        }
    }

    if constexpr (b_mds_t::rank() == 1) {
        // initialize out
        for (index_t idx = 0; idx < n; idx++) {
            x_mds(p_indices(idx)) = b_mds(idx);
        }

        // forward substitution
        for (index_t idx = 0; idx < n; idx++) {
            for (index_t jdx = 0; jdx < idx; jdx++) {
                x_mds(idx) -= l(idx, jdx) * x_mds(jdx);
            }
        }

        // backward substitution
        for (index_t i = n; i > 0; i--) {
            const index_t idx = i - 1;

            for (index_t jdx = i; jdx < n; jdx++) {
                x_mds(idx) -= u(idx, jdx) * x_mds(jdx);
            }

            if (u(idx, idx) == 0) {
                return false;
            }

            x_mds(idx) /= u(idx, idx);
        }

    } else {
        const index_t nrhs = b_mds.extent(1);

        assert(x_mds.extent(1) == nrhs);

        for (index_t rhs = 0; rhs < nrhs; rhs++) {
            // initialize out
            for (index_t idx = 0; idx < n; idx++) {
                x_mds(p_indices(idx), rhs) = b_mds(idx, rhs);
            }

            // forward substitution
            for (index_t idx = 0; idx < n; idx++) {
                for (index_t jdx = 0; jdx < idx; jdx++) {
                    x_mds(idx, rhs) -= l(idx, jdx) * x_mds(jdx, rhs);
                }
            }

            // backward substitution
            for (index_t i = n; i > 0; i--) {
                const index_t idx = i - 1;

                for (index_t jdx = i; jdx < n; jdx++) {
                    x_mds(idx, rhs) -= u(idx, jdx) * x_mds(jdx, rhs);
                }

                if (u(idx, idx) == 0) {
                    return false;
                }

                x_mds(idx, rhs) /= u(idx, idx);
            }
        }
    }

    return true;
}

} // namespace detail

template <MPMode mpmode = MPMode::NONE, typename a_t, typename b_t,
          typename x_t, typename valid_t>
inline constexpr void solve_to(a_t &&a, b_t &&b, x_t &&x, valid_t &&valid) {
    const auto a_mds = core::to_const_mdspan(std::forward<a_t>(a));
    const auto b_mds = core::to_const_mdspan(std::forward<b_t>(b));

    constexpr size_t rhs_rank = decltype(b_mds)::rank() == 1 ? 1 : 2;

    core::batch<mpmode>(
        [](auto &&a, auto &&b, auto &&x, auto &&valid) {
            valid() = detail::solve_impl(std::forward<decltype(a)>(a),
                                         std::forward<decltype(b)>(b),
                                         std::forward<decltype(x)>(x));
        },
        std::index_sequence<2, rhs_rank, rhs_rank, 0>{}, a_mds, b_mds,
        core::to_mdspan(std::forward<x_t>(x)),
        core::to_mdspan(std::forward<valid_t>(valid)));
}

template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename a_t,
          typename b_t>
[[nodiscard]] inline constexpr auto solve(a_t &&a, b_t &&b) {
    const auto a_mds = core::to_const_mdspan(std::forward<a_t>(a));
    const auto b_mds = core::to_const_mdspan(std::forward<b_t>(b));

    constexpr size_t rhs_rank = decltype(b_mds)::rank() == 1 ? 1 : 2;

    auto x = core::create_out<dtype>(
        std::index_sequence<2, rhs_rank>{},
        core::slice_from_right<rhs_rank>(b_mds.extents()), a_mds, b_mds);

    auto valid = core::create_out<uint8_t>(std::index_sequence<2, rhs_rank>{},
                                           extents<uint8_t>{}, a_mds, b_mds);

    solve_to<mpmode>(a_mds, b_mds, x, valid);

    return std::pair{x, valid};
}

} // namespace linalg
} // namespace mdtensor
