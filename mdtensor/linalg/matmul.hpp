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

#include "../manipulation/expand_dims.hpp"
#include "matvec.hpp"
#include "vecmat.hpp"

#ifdef MDTENSOR_USE_EIGEN
#include "../core/eigen/eigen.hpp"
#endif

// TODO: modifiy

namespace mdtensor::linalg {
namespace ufunc {

constexpr void matmul_ufunc_native_noalias(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using out_index_t = typename decltype(out_mds)::index_type;
    using in1_index_t = typename decltype(in1_mds)::index_type;

    for (out_index_t i = 0; i < out_mds.extent(0); i++) {
        for (out_index_t j = 0; j < out_mds.extent(1); j++) {
            out_mds(i, j) = 0;

            for (in1_index_t k = 0; k < in1_mds.extent(1); k++) {
                out_mds(i, j) += in1_mds(i, k) * in2_mds(k, j);
            }
        }
    }
}

constexpr void matmul_ufunc_native(auto &&in1, auto &&in2, auto &&out) {
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
        matmul_ufunc_native_noalias(in1_mds, in2_mds, out_mds);

    } else {
        auto out_tmp = empty_like(out_mds);
        matmul_ufunc_native_noalias(in1_mds, in2_mds, out_tmp);
        static_cast<void>(copy(out_tmp, out_mds));
    }
}

#ifdef MDTENSOR_USE_EIGEN

template <core::mdspan_c in1_t, core::mdspan_c in2_t, core::mdspan_c out_t>
    requires(core::eigen::eigen_mappable_c<in1_t> &&
             core::eigen::eigen_mappable_c<in2_t> &&
             core::eigen::eigen_mappable_c<out_t>)
inline void matmul_ufunc_eigen(const in1_t &in1, const in2_t &in2,
                               const out_t &out) {
    using value_t = core::common_arithmetic_type_t<typename in1_t::value_type,
                                                   typename in2_t::value_type>;

    const auto ein1 = core::eigen::to_eigen(in1);
    const auto ein2 = core::eigen::to_eigen(in2);
    auto eout = core::eigen::to_eigen(out);

    eout = (ein1.template cast<value_t>() * ein2.template cast<value_t>())
               .template cast<typename out_t::value_type>();
}

#endif

constexpr core::Backend matmul_auto_backend(auto &&in1, auto &&in2,
                                            auto &&out) {
    if (std::is_constant_evaluated()) {
        return core::Backend::NATIVE;
    }

#ifdef MDTENSOR_USE_EIGEN
    if constexpr (core::eigen::eigen_mappable_c<decltype(in1)> &&
                  core::eigen::eigen_mappable_c<decltype(in2)> &&
                  core::eigen::eigen_mappable_c<decltype(out)>) {
        return core::Backend::EIGEN;
    }
#endif

    return core::Backend::NATIVE;
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void matmul_to(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    constexpr bool is_in1_mds_1d = (in1_mds.rank() == 1);
    constexpr bool is_in2_mds_1d = (in2_mds.rank() == 1);

    if constexpr (is_in1_mds_1d && !is_in2_mds_1d) {
        vecmat_to<backend>(in1_mds, in2_mds, out_mds);

    } else if constexpr (!is_in1_mds_1d && is_in2_mds_1d) {
        matvec_to<backend>(in1_mds, in2_mds, out_mds);

    } else {
        const auto be = backend;
        // constexpr auto be =
        //     (backend == core::Backend::AUTO)
        //         ?
        // ufunc::matmul_auto_backend(std::forward<decltype(in1)>(in1),
        // std::forward<decltype(in2)>(in2),
        // std::forward<decltype(out)>(out))
        //         : backend;

        if (
#ifdef MDTENSOR_USE_EIGEN
            be == core::Backend::EIGEN
#else
            false
#endif
        ) {
#ifdef MDTENSOR_USE_EIGEN
            core::batch_with_broadcast<core::Backend::NATIVE>(
                [](auto &&...elems) {
                    ufunc::matmul_ufunc_eigen(
                        std::forward<decltype(elems)>(elems)...);
                },
                std::index_sequence<2, 2, 2>{},
                std::integer_sequence<bool, true, true, false>{},
                std::forward<decltype(in1)>(in1),
                std::forward<decltype(in2)>(in2),
                std::forward<decltype(out)>(out));
#endif

        } else if (be == core::Backend::NATIVE) {
            core::batch_with_broadcast<core::Backend::NATIVE>(
                [](auto &&...elems) {
                    ufunc::matmul_ufunc_native(
                        std::forward<decltype(elems)>(elems)...);
                },
                std::index_sequence<2, 2, 2>{},
                std::integer_sequence<bool, true, true, false>{},
                std::forward<decltype(in1)>(in1),
                std::forward<decltype(in2)>(in2),
                std::forward<decltype(out)>(out));

        } else {
            core::batch_with_broadcast<core::Backend::NATIVE>(
                [](auto &&...elems) {
                    ufunc::matmul_ufunc_native(
                        std::forward<decltype(elems)>(elems)...);
                },
                std::index_sequence<2, 2, 2>{},
                std::integer_sequence<bool, true, true, false>{},
                std::forward<decltype(in1)>(in1),
                std::forward<decltype(in2)>(in2),
                std::forward<decltype(out)>(out));
        }
    }
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto matmul(auto &&in1, auto &&in2) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    constexpr bool is_in1_mds_1d = (in1_mds.rank() == 1);
    constexpr bool is_in2_mds_1d = (in2_mds.rank() == 1);

    if constexpr (is_in1_mds_1d && !is_in2_mds_1d) {
        return vecmat<dtype, backend>(in1_mds, in2_mds);

    } else if constexpr (!is_in1_mds_1d && is_in2_mds_1d) {
        return matvec<dtype, backend>(in1_mds, in2_mds);

    } else {
        const auto uin1_exts =
            core::slice_extents_from_right<2>(in1_mds.extents());
        const auto uin2_exts =
            core::slice_extents_from_right<2>(in2_mds.extents());
        const auto uout_exts =
            core::compose_extents(core::slice_extents_from_left<1>(uin1_exts),
                                  core::slice_extents_from_right<1>(uin2_exts));

        auto out = core::make_broadcasted_tensor<dtype>(
            std::index_sequence<uin1_exts.rank(), uin2_exts.rank()>{},
            uout_exts, in1_mds, in2_mds);

        matmul_to<backend>(in1_mds, in2_mds, out);

        return out;
    }
}

} // namespace mdtensor::linalg
