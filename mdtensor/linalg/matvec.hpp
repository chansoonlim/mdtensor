/**
 * @file
 * @brief Matrix-vector multiplication utilities for mdtensor (linalg).
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
inline constexpr void matvec_naive(in1_t &&in1, in2_t &&in2,
                                   out_t &&out) noexcept {
    static_assert(std::remove_cvref_t<in1_t>::rank() == 2);
    static_assert(std::remove_cvref_t<in2_t>::rank() == 1);
    static_assert(std::remove_cvref_t<out_t>::rank() == 1);

    bool need_copy = false;

    if constexpr (requires {
                      core::to_const_mdspan(std::forward<in1_t>(in1))
                              .data_handle() ==
                          core::to_const_mdspan(std::forward<out_t>(out))
                              .data_handle();
                  }) {
        if (core::to_const_mdspan(std::forward<in1_t>(in1)).data_handle() ==
            core::to_const_mdspan(std::forward<out_t>(out)).data_handle())
            [[unlikely]] {
            need_copy = true;
        }
    }

    if constexpr (requires {
                      core::to_const_mdspan(std::forward<in2_t>(in2))
                              .data_handle() ==
                          core::to_const_mdspan(std::forward<out_t>(out))
                              .data_handle();
                  }) {
        if (core::to_const_mdspan(std::forward<in2_t>(in2)).data_handle() ==
            core::to_const_mdspan(std::forward<out_t>(out)).data_handle())
            [[unlikely]] {
            need_copy = true;
        }
    }

    if (!need_copy) [[likely]] {
        using out_index_t = typename std::remove_cvref_t<out_t>::index_type;
        using in1_index_t = typename std::remove_cvref_t<in1_t>::index_type;

        for (out_index_t i = 0; i < out.extent(0); i++) {
            out(i) = 0;
            for (in1_index_t j = 0; j < in1.extent(1); j++) {
                out(i) += in1(i, j) * in2(j);
            }
        }

    } else [[unlikely]] {
        auto out_tmp = empty_like(std::forward<out_t>(out));
        matvec_naive(in1, in2, out_tmp.to_mdspan());
        copy_to(out_tmp, out);
    }
}

template <md_c in1_t, md_c in2_t, md_c out_t>
inline constexpr void matvec_impl(in1_t &&in1, in2_t &&in2,
                                  out_t &&out) noexcept {
    static_assert(std::remove_cvref_t<in1_t>::rank() == 2);
    static_assert(std::remove_cvref_t<in2_t>::rank() == 1);
    static_assert(std::remove_cvref_t<out_t>::rank() == 1);

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

    matvec_naive(in1, in2, out);
}

} // namespace detail

/**
 * @brief Matrix-vector multiplication (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 Input matrix (mdspan, mdarray, etc.) (... x M x K).
 * @param in2 Input vector (mdspan, mdarray, etc.) (... x K).
 * @param out Output vector (mdspan, mdarray, etc.) (... x M).
 *
 * @note The multiplication is performed per instance when inputs are batched.
 *       The last 2 axes of `in1` are treated as matrix axes, and the last axis
 *       of `in2` / `out` is treated as the vector axis.
 *
 * @see mdtensor::linalg::matvec for the out-of-place version that returns the
 * result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void matvec_to(in1_t &&in1, in2_t &&in2,
                                out_t &&out) noexcept {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::matvec_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 1, 1>{},
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Matrix-vector multiplication (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 * inputs.
 *
 * @param in1 Input matrix (mdspan, mdarray, etc.) (... x M x K).
 * @param in2 Input vector (mdspan, mdarray, etc.) (... x K).
 *
 * @return Vector (mdarray) with shape (... x M).
 *
 * @note The multiplication is performed per instance when inputs are batched.
 *
 * @see mdtensor::linalg::matvec_to for the in-place version that writes into an
 * output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto matvec(in1_t &&in1, in2_t &&in2) noexcept {
    const auto in1_mds = core::to_const_mdspan(std::forward<in1_t>(in1));
    const auto in2_mds = core::to_const_mdspan(std::forward<in2_t>(in2));

    const auto uin1_exts = core::slice_from_right<2>(in1_mds.extents());
    const auto uin2_exts = core::slice_from_right<2>(in2_mds.extents());
    const auto uout_exts = extents<
        core::common_index_type_t<typename decltype(uin1_exts)::index_type,
                                  typename decltype(uin2_exts)::index_type>,
        decltype(uin1_exts)::static_extent(0)>{uin1_exts.extent(0)};

    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::matvec_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 1>{}, uout_exts, in1_mds, in2_mds);
}

} // namespace linalg

inline constexpr void matvec_to(auto &&...elems) noexcept {
    linalg::matvec_to(std::forward<decltype(elems)>(elems)...);
}

template <typename dtype = void>
[[nodiscard]] inline constexpr auto matvec(auto &&...elems) noexcept {
    return linalg::matvec<dtype>(std::forward<decltype(elems)>(elems)...);
}

} // namespace mdtensor
