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
#include "../core/external/eigen/eigen.hpp"
#endif

namespace mdtensor::linalg {
namespace ufunc {

[[nodiscard]] constexpr bool inv_ufunc_native(auto &&in, auto &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using index_t = typename decltype(in_mds)::index_type;

    auto in_copy = copy(in_mds);
    static_cast<void>(eye(out_mds.extents(), 0, out_mds));

    for (index_t i = 0; i < in_mds.extent(0); i++) {
        index_t pivot_row = i;
        auto max_abs = absolute(in_copy(i, i));

        for (index_t row = i + 1; row < in_mds.extent(0); row++) {
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
            for (index_t j = 0; j < in_mds.extent(0); j++) {
                std::swap(in_copy(i, j), in_copy(pivot_row, j));
                std::swap(out_mds(i, j), out_mds(pivot_row, j));
            }
        }

        const auto pivot = in_copy(i, i);

        for (index_t j = 0; j < in_mds.extent(0); j++) {
            in_copy(i, j) /= pivot;
            out_mds(i, j) /= pivot;
        }

        for (index_t j = 0; j < in_mds.extent(0); j++) {
            if (i == j) {
                continue;
            }

            const auto factor = in_copy(j, i);

            if (factor == 0) {
                continue;
            }

            for (index_t k = 0; k < in_mds.extent(0); k++) {
                in_copy(j, k) -= factor * in_copy(i, k);
                out_mds(j, k) -= factor * out_mds(i, k);
            }
        }
    }

    return true;
}

#ifdef MDTENSOR_USE_EIGEN

[[nodiscard]] constexpr bool inv_ufunc_eigen(auto &&in, auto &&out) {
    const auto ein =
        core::eigen::to_eigen_matrix(std::forward<decltype(in)>(in));
    auto eout = core::eigen::to_eigen_matrix<core::CopyMode::FALSE>(
        std::forward<decltype(out)>(out));

    eout = ein.inverse();

    return true;
}

#endif

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto inv(auto &&in, out_t &&out = out_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using calc_t = core::calc_type_t<dtype, decltype(in_mds)>;

    auto out_md = core::resolve_output_like<calc_t>(
        std::forward<decltype(out)>(out), in_mds);
    auto valid = core::make_broadcasted_tensor<bool>(
        std::index_sequence<2>{}, core::extents<std::uint8_t>{}, in_mds);

    if constexpr (
#ifdef MDTENSOR_USE_EIGEN
        backend == core::Backend::EIGEN
#else
        false
#endif
    ) {
#ifdef MDTENSOR_USE_EIGEN
        core::batch<core::Backend::NATIVE>(
            [](auto &&in, auto &&out, auto &&valid) {
                valid =
                    ufunc::inv_ufunc_eigen(std::forward<decltype(in)>(in),
                                           std::forward<decltype(out)>(out));
            },
            std::index_sequence<2, 2, 0>{},
            std::integer_sequence<bool, true, false, false>{}, in_mds, out_md,
            valid);
#endif

    } else {
        core::batch<backend>(
            [](auto &&in, auto &&out, auto &&valid) {
                valid =
                    ufunc::inv_ufunc_native(std::forward<decltype(in)>(in),
                                            std::forward<decltype(out)>(out));
            },
            std::index_sequence<2, 2, 0>{},
            std::integer_sequence<bool, true, false, false>{}, in_mds, out_md,
            valid);
    }

    return std::pair{out_md, valid};
}

} // namespace mdtensor::linalg
