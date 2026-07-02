/**
 * @file
 * @brief Matrix-matrix multiplication utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../creation/copy.hpp"
#include "../creation/empty_like.hpp"

namespace mdtensor {
namespace linalg {
namespace detail {

template <md_c in1_t, md_c in2_t, md_c out_t>
inline constexpr void matmul_naive_noalias(in1_t &&in1, in2_t &&in2,
                                           out_t &&out) noexcept {
    using out_index_t = typename std::remove_cvref_t<out_t>::index_type;
    using in1_index_t = typename std::remove_cvref_t<in1_t>::index_type;

    for (out_index_t i = 0; i < out.extent(0); i++) {
        for (out_index_t j = 0; j < out.extent(1); j++) {
            out(i, j) = 0;

            for (in1_index_t k = 0; k < in1.extent(1); k++) {
                out(i, j) += in1(i, k) * in2(k, j);
            }
        }
    }
}

template <md_c in1_t, md_c in2_t, md_c out_t>
inline constexpr void matmul_naive(in1_t &&in1, in2_t &&in2,
                                   out_t &&out) noexcept {
    const auto in1_mds = core::to_const_mdspan(std::forward<in1_t>(in1));
    const auto in2_mds = core::to_const_mdspan(std::forward<in2_t>(in2));
    auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    if (std::is_constant_evaluated()) {
        auto out_tmp = empty_like(out_mds);
        matmul_naive_noalias(in1_mds, in2_mds, out_tmp.to_mdspan());
        copy_to(out_tmp, out_mds);
        return;
    }

    bool need_copy = false;

    if constexpr (requires {
                      in1_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in1_mds.data_handle() == out_mds.data_handle()) [[unlikely]] {
            need_copy = true;
        }
    }

    if constexpr (requires {
                      in2_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in2_mds.data_handle() == out_mds.data_handle()) [[unlikely]] {
            need_copy = true;
        }
    }

    if (!need_copy) [[likely]] {
        matmul_naive_noalias(in1_mds, in2_mds, out_mds);

    } else [[unlikely]] {
        auto out_tmp = empty_like(out_mds);
        matmul_naive_noalias(in1_mds, in2_mds, out_tmp.to_mdspan());
        copy_to(out_tmp, out_mds);
    }
}

template <md_c in1_t, md_c in2_t, md_c out_t>
inline constexpr void matmul_impl(in1_t &&in1, in2_t &&in2,
                                  out_t &&out) noexcept {
    static_assert(std::remove_cvref_t<in1_t>::rank() == 2);
    static_assert(std::remove_cvref_t<in2_t>::rank() == 2);
    static_assert(std::remove_cvref_t<out_t>::rank() == 2);

#ifdef MDTENSOR_USE_EIGEN
#if __cplusplus >= 202302L // TODO: Impliement for C++20
    if constexpr (core::eigen::eigen_mappable_mdspan_c<in1_t> &&
                  core::eigen::eigen_mappable_mdspan_c<in2_t> &&
                  core::eigen::eigen_mappable_mdspan_c<out_t>) {
        if (!std::is_constant_evaluated() && 8 <= out.extent(0) + out.extent(1))
            [[likely]] {
            using value_t = core::common_data_type_t<
                typename std::remove_cvref_t<in1_t>::value_type,
                typename std::remove_cvref_t<in2_t>::value_type>;

            const auto ein1 =
                core::eigen::to_eigen(in1).template cast<value_t>();
            const auto ein2 =
                core::eigen::to_eigen(in2).template cast<value_t>();
            auto eout = core::eigen::to_eigen(out);

            eout = (ein1 * ein2)
                       .template cast<
                           typename std::remove_cvref_t<out_t>::value_type>();

            return;
        }
    }

#else
    assert(false && "Eigen inverse not implemented for C++20");

#endif
#endif

    matmul_naive(in1, in2, out);
}

} // namespace detail

/**
 * @brief Matrix multiplication (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input matrix (mdspan, mdarray, etc.) (... x M x K).
 * @param in2 Second input matrix (mdspan, mdarray, etc.) (... x K x N).
 * @param out Output matrix (mdspan, mdarray, etc.) (... x M x N).
 *
 * @note The multiplication is performed per-matrix when inputs are batched.
 *
 * @see mdtensor::linalg::matmul for the out-of-place version that returns the
 * result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void matmul_to(in1_t &&in1, in2_t &&in2,
                                out_t &&out) noexcept {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::matmul_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 2, 2>{},
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Matrix multiplication (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 * input.
 *
 * @param in1 First input matrix (mdspan, mdarray, etc.) (... x M x K).
 * @param in2 Second input matrix (mdspan, mdarray, etc.) (... x K x N).
 *
 * @return matrix (mdarray) matching the batch extents of the inputs.
 *
 * @note The multiplication is performed per-matrix when inputs are batched.
 *
 * @see mdtensor::linalg::matmul_to for the in-place version that writes into an
 * output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto matmul(in1_t &&in1, in2_t &&in2) noexcept {
    const auto in1_mds = core::to_const_mdspan(std::forward<in1_t>(in1));
    const auto in2_mds = core::to_const_mdspan(std::forward<in2_t>(in2));

    const auto uin1_exts = core::slice_from_right<2>(in1_mds.extents());
    const auto uin2_exts = core::slice_from_right<2>(in2_mds.extents());
    const auto uout_exts = extents<
        core::common_index_type_t<typename decltype(uin1_exts)::index_type,
                                  typename decltype(uin2_exts)::index_type>,
        decltype(uin1_exts)::static_extent(0),
        decltype(uin2_exts)::static_extent(1)>{uin1_exts.extent(0),
                                               uin2_exts.extent(1)};

    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::matmul_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 2>{}, uout_exts, in1_mds, in2_mds);
}

} // namespace linalg

inline constexpr void matmul_to(auto &&...elems) noexcept {
    linalg::matmul_to(std::forward<decltype(elems)>(elems)...);
}

template <typename dtype = void>
[[nodiscard]] inline constexpr auto matmul(auto &&...elems) noexcept {
    return linalg::matmul<dtype>(std::forward<decltype(elems)>(elems)...);
}

} // namespace mdtensor
