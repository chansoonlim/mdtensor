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
#include "../creation/empty_like.hpp"
#include "../creation/eye.hpp"
#include "../math/absolute.hpp"

#ifdef MDTENSOR_USE_EIGEN
#include "../core/eigen/eigen.hpp"
#endif

namespace mdtensor::linalg {
namespace ufunc {

[[nodiscard]] constexpr bool inv_native(auto &&in, auto &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using index_t = typename decltype(in_mds)::index_type;

    const index_t n = in_mds.extent(0);

    auto in_copy = copy(in_mds);
    static_cast<void>(eye(out_mds.extents(), 0, out_mds));

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

[[nodiscard]] constexpr bool inv_ufunc(auto &&in, auto &&out) {
#ifdef MDTENSOR_USE_EIGEN
#if __cplusplus >= 202302L // TODO: Impliement for C++20
    if constexpr (core::eigen::eigen_mappable_c<in_t> &&
                  core::eigen::eigen_mappable_c<out_t>) {
        if (!std::is_constant_evaluated()) {
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

    return inv_native(in, out);
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void inv_to(auto &&in, auto &&out, auto &&valid) {
    core::batch_with_broadcast<backend>(
        [](auto &&in, auto &&out, auto &&valid) {
            valid() = ufunc::inv_ufunc(std::forward<decltype(in)>(in),
                                       std::forward<decltype(out)>(out));
        },
        std::index_sequence<2, 2, 0>{},
        std::integer_sequence<bool, true, false, false>{},
        std::forward<decltype(in)>(in), std::forward<decltype(out)>(out),
        std::forward<decltype(valid)>(valid));
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto inv(auto &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out = empty_like<dtype>(in_mds);
    auto valid = core::make_output<bool>(std::index_sequence<2>{},
                                         core::extents<std::uint8_t>{}, in_mds);

    inv_to<backend>(in_mds, out, valid);

    return std::pair{out, valid};
}

} // namespace mdtensor::linalg
