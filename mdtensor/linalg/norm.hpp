/**
 * @file
 * @brief Vector norm (L2) utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../math/multiply.hpp"
#include "../math/sqrt.hpp"
#include "../math/sum.hpp"

namespace mdtensor {
namespace linalg {
namespace detail {

template <md_c in_t, md_c out_t>
inline constexpr void norm_impl(in_t &&in, out_t &&out) noexcept {
    static_assert(std::remove_cvref_t<in_t>::rank() == 1);
    static_assert(std::remove_cvref_t<out_t>::rank() == 0);

    using index_t = typename std::remove_cvref_t<in_t>::index_type;

    out() = 0;
    for (index_t i = 0; i < in.extent(0); i++) {
        out() += in(i)*in(i);
    }
    out() = sqrt(out());
}

} // namespace detail

/**
 * @brief Compute Euclidean (L2) norm of a vector (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input tensor (mdspan, mdarray, etc.) (... x N).
 * @param out Output tensor (mdspan, mdarray, scalar, etc.) (...).
 *
 * @note When `mpmode == MPMode::SIMD`, the implementation uses the element-wise
 *       pipeline `sum(multiply(in, in))` followed by `sqrt` to improve
 *       vectorization opportunities.
 *
 * @see mdtensor::linalg::norm for the out-of-place version that returns the
 * result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void norm_to(in_t &&in, out_t &&out) noexcept {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    if constexpr (mpmode == MPMode::SIMD) [[unlikely]] {
        sum_to<-1, mpmode>(multiply<mpmode>(in_mds, in_mds), out_mds);
        sqrt_to<mpmode>(out_mds, out_mds);

    } else {
        core::batch<mpmode>(
            [](auto &&...elems) {
                detail::norm_impl(std::forward<decltype(elems)>(elems)...);
            },
            std::index_sequence<1, 0>{}, in_mds, out_mds);
    }
}

/**
 * @brief Compute Euclidean (L2) norm of a vector (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 * input.
 *
 * @param in Input tensor (mdspan, mdarray, etc.) (... x N).
 *
 * @return mdarray or scalar matching the batch extents of the input.
 *
 * @note This function creates a rank-0 output container and then calls
 *       mdtensor::linalg::norm_to.
 *
 * @see mdtensor::linalg::norm_to for the in-place version that writes into an
 * output.
 */
template <MPMode mpmode = MPMode::NONE, typename dtype = void, typename in_t>
[[nodiscard]] inline constexpr auto norm(in_t &&in) noexcept {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    auto out = core::create_out<dtype>(std::index_sequence<1>{},
                                       extents<uint8_t>{}, in_mds);

    norm_to<mpmode>(in_mds, out);

    return out;
}

} // namespace linalg
} // namespace mdtensor
