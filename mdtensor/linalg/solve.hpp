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

namespace mdtensor::linalg {
namespace ufunc {

[[nodiscard]] constexpr bool solve_ufunc(auto &&a, auto &&b, auto &&x) {
    const auto a_mds = core::to_const_mdspan(std::forward<decltype(a)>(a));
    const auto b_mds = core::to_const_mdspan(std::forward<decltype(b)>(b));
    const auto x_mds = core::to_output_mdspan(std::forward<decltype(x)>(x));

    using index_t = typename decltype(a_mds)::index_type;

    const index_t n = a_mds.extent(0);

    // LU decomposition of A
    const auto [p_indices, l, u] = lu_p_indices(a_mds);

    // check singularity
    for (index_t idx = 0; idx < n; idx++) {
        if (u(idx, idx) == 0) {
            return false;
        }
    }

    if constexpr (b_mds.rank() == 1) {
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

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void solve_to(auto &&a, auto &&b, auto &&x, auto &&valid) {
    const auto b_mds = core::to_const_mdspan(std::forward<decltype(b)>(b));

    constexpr std::size_t rhs_rank = b_mds.rank() == 1 ? 1 : 2;

    core::batch_with_broadcast<backend>(
        [](auto &&a, auto &&b, auto &&x, auto &&valid) {
            valid() = ufunc::solve_ufunc(std::forward<decltype(a)>(a),
                                         std::forward<decltype(b)>(b),
                                         std::forward<decltype(x)>(x));
        },
        std::index_sequence<2, rhs_rank, rhs_rank, 0>{},
        std::integer_sequence<bool, true, true, false, false>{},
        std::forward<decltype(a)>(a), b_mds, std::forward<decltype(x)>(x),
        std::forward<decltype(valid)>(valid));
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto solve(auto &&a, auto &&b) {
    const auto a_mds = core::to_const_mdspan(std::forward<decltype(a)>(a));
    const auto b_mds = core::to_const_mdspan(std::forward<decltype(b)>(b));

    constexpr std::size_t rhs_rank = b_mds.rank() == 1 ? 1 : 2;

    auto x = core::make_broadcasted_tensor<dtype>(
        std::index_sequence<2, rhs_rank>{},
        core::slice_extents_from_right<rhs_rank>(b_mds.extents()), a_mds,
        b_mds);

    auto valid = core::make_broadcasted_tensor<bool>(
        std::index_sequence<2, rhs_rank>{}, core::extents<std::uint8_t>{},
        a_mds, b_mds);

    solve_to<backend>(a_mds, b_mds, x, valid);

    return std::pair{x, valid};
}

} // namespace mdtensor::linalg
