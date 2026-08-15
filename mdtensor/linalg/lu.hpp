/**
 * @file
 * @brief LU decomposition utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../creation/arange.hpp"
#include "../creation/copy.hpp"
#include "../creation/empty.hpp"
#include "../math/absolute.hpp"

namespace mdtensor::linalg {
namespace ufunc {

constexpr void lu_p_indices_ufunc(auto &&in, auto &&p_indices, auto &&l,
                                  auto &&u) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto p_indices_mds =
        core::to_output_mdspan(std::forward<decltype(p_indices)>(p_indices));
    const auto l_mds = core::to_output_mdspan(std::forward<decltype(l)>(l));
    const auto u_mds = core::to_output_mdspan(std::forward<decltype(u)>(u));

    using in_mds_t = decltype(in_mds);
    using p_indices_mds_t = decltype(p_indices_mds);

    using index_t = typename in_mds_t::index_type;

    constexpr std::size_t m_s = in_mds_t::static_extent(0);

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
    auto row_order = empty<index_t>(core::extents<index_t, m_s>{m});
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

constexpr void lu_full_ufunc(auto &&in, auto &&p, auto &&l, auto &&u) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto p_mds = core::to_output_mdspan(std::forward<decltype(p)>(p));
    const auto l_mds = core::to_output_mdspan(std::forward<decltype(l)>(l));
    const auto u_mds = core::to_output_mdspan(std::forward<decltype(u)>(u));

    using in_mds_t = decltype(in_mds);
    using p_mds_t = decltype(p_mds);

    using index_t = typename p_mds_t::index_type;

    constexpr std::size_t m_s = in_mds_t::static_extent(0);

    const index_t m = in_mds.extent(0);

    auto p_indices = empty<index_t>(core::extents<index_t, m_s>{m});

    lu_p_indices_ufunc(in_mds, p_indices, l_mds, u_mds);

    for (index_t i = 0; i < m; i++) {
        for (index_t j = 0; j < m; j++) {
            p_mds(i, j) = (j == p_indices(i) ? 1 : 0);
        }
    }
}

constexpr void lu_permute_l_ufunc(auto &&in, auto &&pl, auto &&u) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto pl_mds = core::to_output_mdspan(std::forward<decltype(pl)>(pl));
    const auto u_mds = core::to_output_mdspan(std::forward<decltype(u)>(u));

    using in_mds_t = decltype(in_mds);
    using pl_mds_t = decltype(pl_mds);

    using index_t = typename in_mds_t::index_type;
    using value_t = typename pl_mds_t::value_type;

    constexpr std::size_t m_s = in_mds_t::static_extent(0);
    constexpr std::size_t n_s = in_mds_t::static_extent(1);
    constexpr std::size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(0);
    const index_t n = in_mds.extent(1);
    const index_t k = m < n ? m : n;

    auto p_indices = empty<index_t>(core::extents<index_t, m_s>{m});
    auto l = empty<value_t>(core::extents<index_t, m_s, k_s>{m, k});

    lu_p_indices_ufunc(in_mds, p_indices, l, u_mds);

    // Apply the permutation to L
    for (index_t i = 0; i < m; i++) {
        for (index_t j = 0; j < k; j++) {
            pl_mds(i, j) = l(p_indices(i), j);
        }
    }
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_tuple_t = std::nullopt_t>
[[nodiscard]] constexpr auto
lu_p_indices(auto &&in, out_tuple_t &&out_tuple = out_tuple_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using in_mds_t = decltype(in_mds);
    using index_t = typename in_mds_t::index_type;

    constexpr std::size_t rank = in_mds_t::rank();

    constexpr std::size_t m_s = in_mds_t::static_extent(rank - 2);
    constexpr std::size_t n_s = in_mds_t::static_extent(rank - 1);
    constexpr std::size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(rank - 2);
    const index_t n = in_mds.extent(rank - 1);
    const index_t k = m < n ? m : n;

    using calc_t = core::calc_type_t<dtype, decltype(in_mds)>;

    auto [p_indices_md, l_md, u_md] = core::resolve_broadcasted_outputs<calc_t>(
        std::forward<decltype(out_tuple)>(out_tuple), std::index_sequence<2>{},
        std::tuple{core::extents<index_t, m_s>{m},
                   core::extents<index_t, m_s, k_s>{m, k},
                   core::extents<index_t, k_s, n_s>{k, n}},
        in_mds);

    core::batch<backend>(
        [](auto &&...elems) {
            ufunc::lu_p_indices_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 1, 2, 2>{},
        std::integer_sequence<bool, true, false, false, false>{}, in_mds,
        p_indices_md, l_md, u_md);

    return std::tuple{p_indices_md, l_md, u_md};
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_tuple_t = std::nullopt_t>
[[nodiscard]] constexpr auto
lu_full(auto &&in, out_tuple_t &&out_tuple = out_tuple_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using in_mds_t = decltype(in_mds);
    using index_t = typename in_mds_t::index_type;

    constexpr std::size_t rank = in_mds_t::rank();

    constexpr std::size_t m_s = in_mds_t::static_extent(rank - 2);
    constexpr std::size_t n_s = in_mds_t::static_extent(rank - 1);
    constexpr std::size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(rank - 2);
    const index_t n = in_mds.extent(rank - 1);
    const index_t k = m < n ? m : n;

    using calc_t = core::calc_type_t<dtype, decltype(in_mds)>;

    auto [p_md, l_md, u_md] = core::resolve_broadcasted_outputs<calc_t>(
        std::forward<decltype(out_tuple)>(out_tuple), std::index_sequence<2>{},
        std::tuple{core::extents<index_t, m_s, m_s>{m, m},
                   core::extents<index_t, m_s, k_s>{m, k},
                   core::extents<index_t, k_s, n_s>{k, n}},
        in_mds);

    core::batch<backend>(
        [](auto &&...elems) {
            ufunc::lu_full_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 2, 2, 2>{},
        std::integer_sequence<bool, true, false, false, false>{}, in_mds, p_md,
        l_md, u_md);

    return std::tuple{p_md, l_md, u_md};
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_tuple_t = std::nullopt_t>
[[nodiscard]] constexpr auto
lu_permute_l(auto &&in, out_tuple_t &&out_tuple = out_tuple_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using in_mds_t = decltype(in_mds);
    using index_t = typename in_mds_t::index_type;

    constexpr std::size_t rank = in_mds_t::rank();

    constexpr std::size_t m_s = in_mds_t::static_extent(rank - 2);
    constexpr std::size_t n_s = in_mds_t::static_extent(rank - 1);
    constexpr std::size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(rank - 2);
    const index_t n = in_mds.extent(rank - 1);
    const index_t k = m < n ? m : n;

    using calc_t = core::calc_type_t<dtype, decltype(in_mds)>;

    auto [pl_md, u_md] = core::resolve_broadcasted_outputs<calc_t>(
        std::forward<decltype(out_tuple)>(out_tuple), std::index_sequence<2>{},
        std::tuple{core::extents<index_t, m_s, k_s>{m, k},
                   core::extents<index_t, k_s, n_s>{k, n}},
        in_mds);

    core::batch<backend>(
        [](auto &&...elems) {
            ufunc::lu_permute_l_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 2, 2>{},
        std::integer_sequence<bool, true, false, false>{}, in_mds, pl_md, u_md);

    return std::tuple{pl_md, u_md};
}

template <bool permute_l = false, bool p_indices = false, typename dtype = void,
          core::Backend backend = core::Backend::AUTO,
          typename out_tuple_t = std::nullopt_t>
[[nodiscard]] constexpr auto
lu(auto &&in, out_tuple_t &&out_tuple = out_tuple_t{std::nullopt}) {
    static_assert(!(permute_l && p_indices),
                  "lu cannot return both permuted L and P indices.");

    if constexpr (permute_l) {
        return lu_permute_l<dtype, backend>(
            std::forward<decltype(in)>(in),
            std::forward<decltype(out_tuple)>(out_tuple));

    } else if constexpr (p_indices) {
        return lu_p_indices<dtype, backend>(
            std::forward<decltype(in)>(in),
            std::forward<decltype(out_tuple)>(out_tuple));

    } else {
        return lu_full<dtype, backend>(
            std::forward<decltype(in)>(in),
            std::forward<decltype(out_tuple)>(out_tuple));
    }
}

} // namespace mdtensor::linalg
