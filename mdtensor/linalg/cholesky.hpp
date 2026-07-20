/**
 * @file
 * @brief Linear system solve utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../creation/empty_like.hpp"
#include "../creation/fill.hpp"
#include "../math/sqrt.hpp"

namespace mdtensor {
namespace linalg {
namespace detail {

template <md_c in_t, md_c out_t>
[[nodiscard]] inline constexpr bool cholesky_upper_impl(in_t &&in,
                                                        out_t &&out) {
    auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using out_mds_t = std::remove_cvref_t<decltype(out_mds)>;

    using index_t = typename in_mds_t::index_type;
    using value_t = typename out_mds_t::value_type;

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

template <md_c in_t, md_c out_t>
[[nodiscard]] inline constexpr bool cholesky_lower_impl(in_t &&in,
                                                        out_t &&out) {
    auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using out_mds_t = std::remove_cvref_t<decltype(out_mds)>;

    using index_t = typename in_mds_t::index_type;
    using value_t = typename out_mds_t::value_type;

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

template <bool upper, md_c in_t, md_c out_t>
[[nodiscard]] inline constexpr bool cholesky_impl(in_t &&in, out_t &&out) {
    if constexpr (upper) {
        return cholesky_upper_impl(std::forward<in_t>(in),
                                   std::forward<out_t>(out));

    } else {
        return cholesky_lower_impl(std::forward<in_t>(in),
                                   std::forward<out_t>(out));
    }
}

} // namespace detail

template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t,
          typename valid_t>
inline constexpr void cholesky_to(in_t &&in, out_t &&out, valid_t &&valid,
                                  const bool upper = false) {
    const auto run_batch = [&]<bool upper_v>() {
        core::batch<mpmode>(
            [](auto &&in, auto &&out, auto &&valid) {
                valid() = detail::cholesky_impl<upper_v>(
                    std::forward<decltype(in)>(in),
                    std::forward<decltype(out)>(out));
            },
            std::index_sequence<2, 2, 0>{},
            core::to_const_mdspan(std::forward<in_t>(in)),
            core::to_mdspan(std::forward<out_t>(out)),
            core::to_mdspan(std::forward<valid_t>(valid)));
    };

    if (upper) {
        run_batch.template operator()<true>();

    } else {
        run_batch.template operator()<false>();
    }
}

template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto cholesky(in_t &&in,
                                             const bool upper = false) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    auto out = empty_like(in_mds);
    auto valid = core::create_out<bool>(std::index_sequence<2>{},
                                        extents<uint8_t>{}, in_mds);

    cholesky_to<mpmode>(in_mds, out, valid, upper);

    return std::pair{out, valid};
}

} // namespace linalg
} // namespace mdtensor
