/**
 * @file
 * @brief SciPy-like LU decomposition utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../creation/arange.hpp"
#include "../creation/copy.hpp"
#include "../math/absolute.hpp"

namespace mdtensor {
namespace linalg {
namespace detail {

template <core::md_c in_t, core::md_c p_indices_t, core::md_c l_t,
          core::md_c u_t>
inline constexpr void lu_p_indices_impl(in_t &&in, p_indices_t &&p_indices,
                                        l_t &&l, u_t &&u) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto p_indices_mds =
        core::to_mdspan(std::forward<p_indices_t>(p_indices));
    const auto l_mds = core::to_mdspan(std::forward<l_t>(l));
    const auto u_mds = core::to_mdspan(std::forward<u_t>(u));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using p_indices_mds_t = std::remove_cvref_t<decltype(p_indices_mds)>;
    using l_mds_t = std::remove_cvref_t<decltype(l_mds)>;
    using u_mds_t = std::remove_cvref_t<decltype(u_mds)>;

    static_assert(in_mds_t::rank() == 2);
    static_assert(p_indices_mds_t::rank() == 1);
    static_assert(l_mds_t::rank() == 2);
    static_assert(u_mds_t::rank() == 2);

    using index_t = typename in_mds_t::index_type;

    constexpr size_t m_s = in_mds_t::static_extent(0);

    const index_t m = in_mds.extent(0);
    const index_t n = in_mds.extent(1);
    const index_t k = m < n ? m : n;

    assert(p_indices_mds.extent(0) == m);
    assert(l_mds.extent(0) == m);
    assert(l_mds.extent(1) == k);
    assert(u_mds.extent(0) == k);
    assert(u_mds.extent(1) == n);

    // initialize
    auto in_copy = copy(in_mds);
    auto row_order =
        core::make_container<index_t>(core::stdex::extents<index_t, m_s>{m});
    for (index_t i = 0; i < m; i++) {
        row_order(i) = i;
    }

    // Compute LU decomposition with partial pivoting
    for (index_t i = 0; i < k; i++) {
        // find maximum element in the current column
        index_t pivot_row = i;
        auto max_abs = absolute(in_copy(i, i));

        for (index_t j = i + 1; j < m; j++) {
            const auto candidate = absolute(in_copy(j, i));
            if (candidate > max_abs) {
                max_abs = candidate;
                pivot_row = j;
            }
        }

        // swap row if necessary
        if (pivot_row != i) {
            for (index_t j = 0; j < n; j++) {
                std::swap(in_copy(i, j), in_copy(pivot_row, j));
            }
            std::swap(row_order(i), row_order(pivot_row));
        }

        // no multiplier can be formed
        if (max_abs == 0) {
            continue;
        }

        // compute the multipliers and update the U matrix
        for (index_t j = i + 1; j < m; j++) {
            in_copy(j, i) /= in_copy(i, i);

            const auto factor = in_copy(j, i);

            for (index_t k = i + 1; k < n; k++) {
                in_copy(j, k) -= factor * in_copy(i, k);
            }
        }
    }

    // Generate P
    for (index_t i = 0; i < m; i++) {
        p_indices_mds(row_order(i)) =
            static_cast<typename p_indices_mds_t::value_type>(i);
    }

    // Generate L
    for (index_t i = 0; i < m; i++) {
        for (index_t j = 0; j < k; j++) {
            if (i > j) {
                l_mds(i, j) = in_copy(i, j);

            } else if (i == j) {
                l_mds(i, j) = 1;

            } else {
                l_mds(i, j) = 0;
            }
        }
    }

    // Generate U
    for (index_t i = 0; i < k; i++) {
        for (index_t j = 0; j < n; j++) {
            if (i <= j) {
                u_mds(i, j) = in_copy(i, j);

            } else {
                u_mds(i, j) = 0;
            }
        }
    }
}

template <core::md_c in_t, core::md_c p_t, core::md_c l_t, core::md_c u_t>
inline constexpr void lu_full_impl(in_t &&in, p_t &&p, l_t &&l, u_t &&u) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto p_mds = core::to_mdspan(std::forward<p_t>(p));
    const auto l_mds = core::to_mdspan(std::forward<l_t>(l));
    const auto u_mds = core::to_mdspan(std::forward<u_t>(u));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using p_mds_t = std::remove_cvref_t<decltype(p_mds)>;
    using l_mds_t = std::remove_cvref_t<decltype(l_mds)>;
    using u_mds_t = std::remove_cvref_t<decltype(u_mds)>;

    static_assert(in_mds_t::rank() == 2);
    static_assert(p_mds_t::rank() == 2);
    static_assert(l_mds_t::rank() == 2);
    static_assert(u_mds_t::rank() == 2);

    using index_t = typename p_mds_t::index_type;

    constexpr size_t m_s = in_mds_t::static_extent(0);

    const index_t m = in_mds.extent(0);

    auto p_indices =
        core::make_container<index_t>(core::stdex::extents<index_t, m_s>{m});

    lu_p_indices_impl(in_mds, p_indices, l_mds, u_mds);

    for (index_t i = 0; i < m; i++) {
        for (index_t j = 0; j < m; j++) {
            p_mds(i, j) = (j == p_indices(i) ? 1 : 0);
        }
    }
}

template <core::md_c in_t, core::md_c pl_t, core::md_c u_t>
inline constexpr void lu_permute_l_impl(in_t &&in, pl_t &&pl, u_t &&u) {
    auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    auto pl_mds = core::to_mdspan(std::forward<pl_t>(pl));
    auto u_mds = core::to_mdspan(std::forward<u_t>(u));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using pl_mds_t = std::remove_cvref_t<decltype(pl_mds)>;
    using u_mds_t = std::remove_cvref_t<decltype(u_mds)>;

    static_assert(in_mds_t::rank() == 2);
    static_assert(pl_mds_t::rank() == 2);
    static_assert(u_mds_t::rank() == 2);

    using index_t = typename in_mds_t::index_type;
    using value_t = typename pl_mds_t::value_type;

    constexpr size_t m_s = in_mds_t::static_extent(0);
    constexpr size_t n_s = in_mds_t::static_extent(1);
    constexpr size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(0);
    const index_t n = in_mds.extent(1);
    const index_t k = m < n ? m : n;

    auto p_indices =
        core::make_container<index_t>(core::stdex::extents<index_t, m_s>{m});
    auto l = core::make_container<value_t>(
        core::stdex::extents<index_t, m_s, k_s>{m, k});

    lu_p_indices_impl(in_mds, p_indices, l, u_mds);

    // Apply the permutation to L
    for (index_t i = 0; i < m; i++) {
        for (index_t j = 0; j < k; j++) {
            pl_mds(i, j) = l(p_indices(i), j);
        }
    }
}

} // namespace detail

template <core::MPMode mpmode = core::MPMode::NONE, typename in_t,
          typename p_indices_t, typename l_t, typename u_t>
inline constexpr void lu_p_indices_to(in_t &&in, p_indices_t &&p_indices,
                                      l_t &&l, u_t &&u) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::lu_p_indices_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 1, 2, 2>{},
        std::integer_sequence<bool, false, true, true, true>{},
        std::forward<in_t>(in), std::forward<p_indices_t>(p_indices),
        std::forward<l_t>(l), std::forward<u_t>(u));
}

template <core::MPMode mpmode = core::MPMode::NONE, typename in_t, typename p_t,
          typename l_t, typename u_t>
inline constexpr void lu_full_to(in_t &&in, p_t &&p, l_t &&l, u_t &&u) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::lu_full_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 2, 2, 2>{},
        std::integer_sequence<bool, false, true, true, true>{},
        std::forward<in_t>(in), std::forward<p_t>(p), std::forward<l_t>(l),
        std::forward<u_t>(u));
}

template <core::MPMode mpmode = core::MPMode::NONE, typename in_t,
          typename pl_t, typename u_t>
inline constexpr void lu_permute_l_to(in_t &&in, pl_t &&pl, u_t &&u) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::lu_permute_l_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 2, 2>{},
        std::integer_sequence<bool, false, true, true>{},
        std::forward<in_t>(in), std::forward<pl_t>(pl), std::forward<u_t>(u));
}

template <typename dtype = void, core::MPMode mpmode = core::MPMode::NONE,
          typename in_t>
[[nodiscard]] inline constexpr auto lu_p_indices(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;

    using index_t = typename in_mds_t::index_type;

    constexpr size_t rank = in_mds_t::rank();

    constexpr size_t m_s = in_mds_t::static_extent(rank - 2);
    constexpr size_t n_s = in_mds_t::static_extent(rank - 1);
    constexpr size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(rank - 2);
    const index_t n = in_mds.extent(rank - 1);
    const index_t k = m < n ? m : n;

    auto outs = core::create_outs<dtype>(
        std::index_sequence<2>{},
        std::tuple{extents<index_t, m_s>{m},
                   core::stdex::extents<index_t, m_s, k_s>{m, k},
                   core::stdex::extents<index_t, k_s, n_s>{k, n}},
        in_mds);

    lu_p_indices_to<mpmode>(in_mds, std::get<0>(outs), std::get<1>(outs),
                            std::get<2>(outs));

    return outs;
}

template <typename dtype = void, core::MPMode mpmode = core::MPMode::NONE,
          typename in_t>
[[nodiscard]] inline constexpr auto lu_full(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;

    using index_t = typename in_mds_t::index_type;

    constexpr size_t rank = in_mds_t::rank();

    constexpr size_t m_s = in_mds_t::static_extent(rank - 2);
    constexpr size_t n_s = in_mds_t::static_extent(rank - 1);
    constexpr size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(rank - 2);
    const index_t n = in_mds.extent(rank - 1);
    const index_t k = m < n ? m : n;

    auto outs = core::create_outs<dtype>(
        std::index_sequence<2>{},
        std::tuple{extents<index_t, m_s, m_s>{m, m},
                   core::stdex::extents<index_t, m_s, k_s>{m, k},
                   core::stdex::extents<index_t, k_s, n_s>{k, n}},
        in_mds);

    lu_full_to<mpmode>(in_mds, std::get<0>(outs), std::get<1>(outs),
                       std::get<2>(outs));

    return outs;
}

template <typename dtype = void, core::MPMode mpmode = core::MPMode::NONE,
          typename in_t>
[[nodiscard]] inline constexpr auto lu_permute_l(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;

    using index_t = typename in_mds_t::index_type;

    constexpr size_t rank = in_mds_t::rank();

    constexpr size_t m_s = in_mds_t::static_extent(rank - 2);
    constexpr size_t n_s = in_mds_t::static_extent(rank - 1);
    constexpr size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(rank - 2);
    const index_t n = in_mds.extent(rank - 1);
    const index_t k = m < n ? m : n;

    auto outs = core::create_outs<dtype>(
        std::index_sequence<2>{},
        std::tuple{extents<index_t, m_s, k_s>{m, k},
                   core::stdex::extents<index_t, k_s, n_s>{k, n}},
        in_mds);

    lu_permute_l_to<mpmode>(in_mds, std::get<0>(outs), std::get<1>(outs));

    return outs;
}

template <bool permute_l = false, bool p_indices = false, typename dtype = void,
          core::MPMode mpmode = core::MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto lu(in_t &&in) {
    static_assert(!(permute_l && p_indices),
                  "lu cannot return both permuted L and P indices.");

    if constexpr (permute_l) {
        return lu_permute_l<dtype, mpmode>(std::forward<in_t>(in));

    } else if constexpr (p_indices) {
        return lu_p_indices<dtype, mpmode>(std::forward<in_t>(in));

    } else {
        return lu_full<dtype, mpmode>(std::forward<in_t>(in));
    }
}

} // namespace linalg
} // namespace mdtensor
