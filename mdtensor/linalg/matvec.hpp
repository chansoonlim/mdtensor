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

#ifdef MDTENSOR_USE_EIGEN
#include "../core/external/eigen/eigen.hpp"
#endif

namespace mdtensor::linalg {
namespace ufunc {

constexpr void matvec_ufunc_native_noalias(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using calc_t =
        core::common_value_type_t<decltype(in1_mds), decltype(in2_mds)>;

    using out_index_t = typename decltype(out_mds)::index_type;
    using in1_index_t = typename decltype(in1_mds)::index_type;

    for (out_index_t i = 0; i < out_mds.extent(0); i++) {
        out_mds(i) = 0;

        for (in1_index_t j = 0; j < in1_mds.extent(1); j++) {
            out_mds(i) += static_cast<calc_t>(in1_mds(i, j)) *
                          static_cast<calc_t>(in2_mds(j));
        }
    }
}

constexpr void matvec_ufunc_native(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    bool need_copy = false;

    if (std::is_constant_evaluated()) {
        need_copy = true;

    } else if ((void *)in1_mds.data_handle() == (void *)out_mds.data_handle() ||
               (void *)in2_mds.data_handle() == (void *)out_mds.data_handle()) {
        need_copy = true;
    }

    if (!need_copy) {
        matvec_ufunc_native_noalias(in1_mds, in2_mds, out_mds);

    } else {
        auto out_tmp = empty_like(out_mds);
        matvec_ufunc_native_noalias(in1_mds, in2_mds, out_tmp);
        static_cast<void>(copy(out_tmp, out_mds));
    }
}

#ifdef MDTENSOR_USE_EIGEN

inline void matvec_ufunc_eigen(auto &&in1, auto &&in2, auto &&out) {
    const auto ein1 = core::eigen::to_eigen_matrix(in1);
    const auto ein2 = core::eigen::to_eigen_vector<true>(in2);
    auto eout = core::eigen::to_eigen_vector<true, core::CopyMode::FALSE>(out);

    using calc_t = core::promote_type_t<typename decltype(ein1)::Scalar,
                                        typename decltype(ein2)::Scalar>;

    eout = (ein1.template cast<calc_t>() * ein2.template cast<calc_t>())
               .template cast<typename decltype(eout)::Scalar>();
}

#endif

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto matvec(auto &&in1, auto &&in2,
                                    out_t &&out = out_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    using calc_t =
        core::calc_type_t<dtype, decltype(in1_mds), decltype(in2_mds)>;

    const auto uout_exts = core::slice_extents_from_left<1>(
        core::slice_extents_from_right<2>(in1_mds.extents()));

    auto out_md = core::resolve_broadcasted_output<calc_t>(
        std::forward<decltype(out)>(out), std::index_sequence<2, 1>{},
        uout_exts, in1_mds, in2_mds);

    if constexpr (
#ifdef MDTENSOR_USE_EIGEN
        backend == core::Backend::EIGEN
#else
        false
#endif
    ) {
#ifdef MDTENSOR_USE_EIGEN
        core::batch<core::Backend::NATIVE>(
            [](auto &&...elems) {
                ufunc::matvec_ufunc_eigen(
                    std::forward<decltype(elems)>(elems)...);
            },
            std::index_sequence<2, 1, 1>{},
            std::integer_sequence<bool, true, true, false>{}, in1_mds, in2_mds,
            out_md);
#endif

    } else {
        core::batch<backend>(
            [](auto &&...elems) {
                ufunc::matvec_ufunc_native(
                    std::forward<decltype(elems)>(elems)...);
            },
            std::index_sequence<2, 1, 1>{},
            std::integer_sequence<bool, true, true, false>{}, in1_mds, in2_mds,
            out_md);
    }

    return out_md;
}

} // namespace mdtensor::linalg
