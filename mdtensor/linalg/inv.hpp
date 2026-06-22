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

namespace mdtensor {
namespace linalg {
namespace detail {

template <md_c in_t, md_c out_t, md_c valid_t>
inline constexpr void inv_naive(in_t &&in, out_t &&out,
                                valid_t &&valid) noexcept {
    static_assert(std::remove_cvref_t<in_t>::rank() == 2);
    static_assert(std::remove_cvref_t<out_t>::rank() == 2);
    static_assert(std::remove_cvref_t<valid_t>::rank() == 0);

    using index_t = typename std::remove_cvref_t<in_t>::index_type;

    const index_t n = in.extent(0);

    auto in_copy = copy(in);

    for (index_t i = 0; i < n; i++) {
        for (index_t j = 0; j < n; j++) {
            out(i, j) = (i == j) ? 1 : 0;
        }
    }

    for (index_t i = 0; i < n; i++) {
        const auto pivot = in_copy(i, i);

        if (pivot == 0) {
            // Handle error: singular matrix (no inverse)
            valid() = false;
            return;
        }

        // Normalize the pivot row
        for (index_t j = 0; j < n; j++) {
            in_copy(i, j) /= pivot;
            out(i, j) /= pivot;
        }

        // Eliminate other rows
        for (index_t j = 0; j < n; j++) {
            if (i == j)
                continue;

            const auto factor = in_copy(j, i);
            for (index_t k = 0; k < n; k++) {
                in_copy(j, k) -= factor * in_copy(i, k);
                out(j, k) -= factor * out(i, k);
            }
        }
    }

    valid() = true;
}

template <md_c in_t, md_c out_t, md_c valid_t>
inline constexpr void inv_impl(in_t &&in, out_t &&out,
                               valid_t &&valid) noexcept {
    static_assert(std::remove_cvref_t<in_t>::rank() == 2);
    static_assert(std::remove_cvref_t<out_t>::rank() == 2);
    static_assert(std::remove_cvref_t<valid_t>::rank() == 0);

#ifdef MDTENSOR_USE_EIGEN
#if __cplusplus >= 202302L // TODO: Impliement for C++20
    if constexpr (core::eigen::eigen_mappable_mdspan_c<in_t> &&
                  core::eigen::eigen_mappable_mdspan_c<out_t>) {
        if (!std::is_constant_evaluated()) [[likely]] {
            const auto ein = core::eigen::to_eigen(in);
            auto eout = core::eigen::to_eigen(out);

            eout = ein.inverse();
            valid() = true; // TODO: Check for invertibility

            return;
        }
    }

#else
    assert(false && "Eigen inverse not implemented for C++20");

#endif
#endif

    inv_naive(in, out, valid);
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
inline constexpr void inv_to(in_t &&in, out_t &&out, valid_t &&valid) noexcept {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::inv_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 2, 0>{},
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<out_t>(out)),
        core::to_mdspan(std::forward<valid_t>(valid)));
}

/**
 * @brief Compute matrix inverse (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 * input.
 *
 * @param in Input matrix (mdspan, mdarray, etc.) (... x N x N).
 *
 * @return A tuple-like output from batch_out consisting of:
 *         - inverse matrix (mdarray) (... x N x N).
 *         - validity flag (mdarray or scalar) (...), indicating if the inverse
 *        was successfully computed for each matrix instance.
 *
 * @note The output layout is determined by the `batch_out` extents tuple passed
 *       to core::batch_out.
 *
 * @see mdtensor::linalg::inv_to for the in-place version that writes into an
 * output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto inv(in_t &&in) noexcept {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::inv_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2>{},
        std::make_tuple(core::slice_from_right<2>(in_mds.extents()),
                        extents<uint8_t>{}),
        in_mds);
}

} // namespace linalg
} // namespace mdtensor
