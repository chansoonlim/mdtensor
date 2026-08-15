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

#ifdef MDTENSOR_USE_EIGEN
#include "../core/external/eigen/eigen.hpp"
#endif

namespace mdtensor::linalg {
namespace ufunc {

[[nodiscard]] constexpr auto norm_ufunc_native(auto &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using value_t = typename decltype(in_mds)::value_type;
    using index_t = typename decltype(in_mds)::index_type;

    value_t sum = 0;

    for (index_t i = 0; i < in_mds.extent(0); i++) {
        sum += in_mds(i) * in_mds(i);
    }

    return sqrt(sum);
}

#ifdef MDTENSOR_USE_EIGEN

[[nodiscard]] constexpr auto norm_ufunc_eigen(auto &&in) {
    const auto ein =
        core::eigen::to_eigen_vector(std::forward<decltype(in)>(in));

    return ein.norm();
}

#endif

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto norm(auto &&in,
                                  out_t &&out = out_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using calc_t = core::calc_type_t<dtype, decltype(in_mds)>;

    auto out_md = core::resolve_broadcasted_output<calc_t>(
        std::forward<decltype(out)>(out), std::index_sequence<1>{},
        core::extents<std::uint8_t>{}, in_mds);

    if constexpr (backend == core::Backend::SIMD) {
        static_cast<void>(sum<void, false, backend>(
            multiply<void, backend>(in_mds, in_mds),
            std::integer_sequence<int, -1>{}, out_md));
        static_cast<void>(sqrt<void, backend>(out_md, out_md));

    } else if constexpr (
#ifdef MDTENSOR_USE_EIGEN
        backend == core::Backend::EIGEN
#else
        false
#endif
    ) {
#ifdef MDTENSOR_USE_EIGEN
        core::batch<core::Backend::NATIVE>(
            [](auto &&in, auto &&out) {
                out = ufunc::norm_ufunc_eigen(std::forward<decltype(in)>(in));
            },
            std::index_sequence<1, 0>{},
            std::integer_sequence<bool, true, false>{}, in_mds, out_md);

#endif

    } else {
        core::batch<backend>(
            [](auto &&in, auto &&out) {
                out = ufunc::norm_ufunc_native(std::forward<decltype(in)>(in));
            },
            std::index_sequence<1, 0>{},
            std::integer_sequence<bool, true, false>{}, in_mds, out_md);
    }

    return out_md;
}

} // namespace mdtensor::linalg
