/**
 * @file
 * @brief Cholesky decomposition utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../creation/empty_like.hpp"
#include "../math/sqrt.hpp"
#include "../util/fill.hpp"

namespace mdtensor::linalg {
namespace ufunc {

[[nodiscard]] constexpr bool cholesky_upper_ufunc(auto &&in, auto &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using index_t = typename decltype(in_mds)::index_type;
    using value_t = typename decltype(out_mds)::value_type;

    const index_t n = in_mds.extent(0);

    // create zero matrix for output
    fill(out_mds, 0);

    // perform the upper Cholesky decomposition
    for (index_t i = 0; i < n; i++) {
        for (index_t j = i; j < n; j++) {
            value_t sum = 0;

            for (index_t k = 0; k < i; k++) {
                sum += out_mds(k, i) * out_mds(k, j);
            }

            if (i == j) {
                const auto diag = in_mds(i, i) - sum;

                if (diag <= 0) {
                    return false;
                }

                out_mds(i, i) = sqrt(diag);

            } else {
                if (out_mds(i, i) <= 0) {
                    return false;
                }

                out_mds(i, j) = (in_mds(i, j) - sum) / out_mds(i, i);
            }
        }
    }

    return true;
}

[[nodiscard]] constexpr bool cholesky_lower_ufunc(auto &&in, auto &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using index_t = typename decltype(in_mds)::index_type;
    using value_t = typename decltype(out_mds)::value_type;

    const index_t n = in_mds.extent(0);

    // create zero matrix for output
    fill(out_mds, 0);

    // perform the cholesky decomposition
    for (index_t i = 0; i < n; i++) {
        for (index_t j = 0; j <= i; j++) {
            value_t sum = 0;

            for (index_t k = 0; k < j; k++) {
                sum += out_mds(i, k) * out_mds(j, k);
            }

            if (i == j) {
                const auto diag = in_mds(i, i) - sum;

                if (diag <= 0) {
                    return false;
                }

                out_mds(i, j) = sqrt(diag);

            } else {
                if (out_mds(j, j) <= 0) {
                    return false;
                }

                out_mds(i, j) = (in_mds(i, j) - sum) / out_mds(j, j);
            }
        }
    }

    return true;
}

template <bool upper>
[[nodiscard]] constexpr bool cholesky_ufunc(auto &&in, auto &&out) {
    if constexpr (upper) {
        return cholesky_upper_ufunc(std::forward<decltype(in)>(in),
                                    std::forward<decltype(out)>(out));

    } else {
        return cholesky_lower_ufunc(std::forward<decltype(in)>(in),
                                    std::forward<decltype(out)>(out));
    }
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void cholesky_to(auto &&in, auto &&out, auto &&valid,
                           const bool upper = false) {
    const auto run_batch = [&]<bool upper_v>() {
        core::batch_with_broadcast<backend>(
            [](auto &&in, auto &&out, auto &&valid) {
                valid() = ufunc::cholesky_ufunc<upper_v>(
                    std::forward<decltype(in)>(in),
                    std::forward<decltype(out)>(out));
            },
            std::index_sequence<2, 2, 0>{},
            std::integer_sequence<bool, true, false, false>{},
            std::forward<decltype(in)>(in), std::forward<decltype(out)>(out),
            std::forward<decltype(valid)>(valid));
    };

    if (upper) {
        run_batch.template operator()<true>();

    } else {
        run_batch.template operator()<false>();
    }
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto cholesky(auto &&in, const bool upper = false) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out = empty_like(in_mds);
    auto valid = core::make_output<bool>(std::index_sequence<2>{},
                                         core::extents<std::uint8_t>{}, in_mds);

    cholesky_to<backend>(in_mds, out, valid, upper);

    return std::pair{out, valid};
}

} // namespace mdtensor::linalg
