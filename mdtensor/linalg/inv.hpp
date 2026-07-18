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

template <md_c in_t, md_c out_t>
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

template <md_c in_t, md_c out_t>
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

/**
 * @brief Compute matrix inverse (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input matrix (mdspan, mdarray, etc.) (... x N x N).
 * @param out Output matrix (mdspan, mdarray, etc.) (... x N x N).
 * @param valid Output mdspan, mdarray, or scalar indicating validity of the
 * inverse.
 *
 * @note The inverse is computed per-matrix when `in` is batched. The `valid`
 *       flag is produced per matrix instance.
 *
 * @see mdtensor::linalg::inv for the out-of-place version that returns the
 * result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t,
          typename valid_t>
inline constexpr void inv_to(in_t &&in, out_t &&out, valid_t &&valid) {
    core::batch<mpmode>(
        [](auto &&in, auto &&out, auto &&valid) {
            valid() = detail::inv_impl(std::forward<decltype(in)>(in),
                                       std::forward<decltype(out)>(out));
        },
        std::index_sequence<2, 2, 0>{},
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<out_t>(out)),
        core::to_mdspan(std::forward<valid_t>(valid)));
}

/**
 * @brief Compute matrix inverse (out-of-place).
 *
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 * input.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input matrix (mdspan, mdarray, etc.) (... x N x N).
 *
 * @return A tuple-like output from batch_out consisting of:
 *         - inverse matrix (mdarray) (... x N x N).
 *         - validity flag (mdarray or scalar) (...), indicating if the inverse
 *        was successfully computed for each matrix instance.
 *
 * @see mdtensor::linalg::inv_to for the in-place version that writes into an
 * output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto inv(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    auto out = core::create_out<dtype>(
        std::index_sequence<2>{}, core::slice_from_right<2>(in_mds.extents()),
        in_mds);
    auto valid = core::create_out<uint8_t>(std::index_sequence<2>{},
                                           extents<uint8_t>{}, in_mds);

    inv_to<mpmode>(in_mds, out, valid);

    return std::pair{out, valid};
}

} // namespace linalg
} // namespace mdtensor
