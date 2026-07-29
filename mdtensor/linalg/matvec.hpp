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
#include "../core/eigen/eigen.hpp"
#endif

namespace mdtensor::linalg {
namespace ufunc {

constexpr void matvec_native_noalias(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using out_index_t = typename decltype(out_mds)::index_type;
    using in1_index_t = typename decltype(in1_mds)::index_type;

    for (out_index_t i = 0; i < out_mds.extent(0); i++) {
        out_mds(i) = 0;

        for (in1_index_t j = 0; j < in1_mds.extent(1); j++) {
            out_mds(i) += in1_mds(i, j) * in2_mds(j);
        }
    }
}

constexpr void matvec_naive(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    if (std::is_constant_evaluated()) {
        auto out_tmp = empty_like(out_mds);
        matvec_native_noalias(in1_mds, in2_mds, out_tmp);
        static_cast<void>(copy(out_tmp, out_mds));
        return;
    }

    bool need_copy = false;

    if constexpr (requires {
                      in1_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in1_mds.data_handle() == out_mds.data_handle()) {
            need_copy = true;
        }
    }

    if constexpr (requires {
                      in2_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in2_mds.data_handle() == out_mds.data_handle()) {
            need_copy = true;
        }
    }

    if (!need_copy) {
        matvec_native_noalias(in1_mds, in2_mds, out_mds);

    } else {
        auto out_tmp = empty_like(out_mds);
        matvec_native_noalias(in1_mds, in2_mds, out_tmp);
        static_cast<void>(copy(out_tmp, out_mds));
    }
}

constexpr void matvec_ufunc(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

#ifdef MDTENSOR_USE_EIGEN
#if __cplusplus >= 202302L // TODO: Impliement for C++20
    using in1_mds_t = decltype(in1_mds);
    using in2_mds_t = decltype(in2_mds);
    using out_mds_t = decltype(out_mds);

    if constexpr (core::eigen::eigen_mappable_c<in1_mds_t> &&
                  core::eigen::eigen_mappable_c<in2_mds_t> &&
                  core::eigen::eigen_mappable_c<out_mds_t>) {
        if (!std::is_constant_evaluated() &&
            8 <= out_mds.extent(0) + out_mds.extent(1)) {
            using value_t = core::common_data_type_t<
                typename std::remove_cvref_t<in1_mds_t>::value_type,
                typename std::remove_cvref_t<in2_mds_t>::value_type>;

            const auto ein1 =
                core::eigen::to_eigen(in1_mds).template cast<value_t>();
            const auto ein2 =
                core::eigen::to_eigen(in2_mds).template cast<value_t>();
            auto eout = core::eigen::to_eigen(out_mds);

            eout =
                (ein1 * ein2)
                    .template cast<
                        typename std::remove_cvref_t<out_mds_t>::value_type>();

            return;
        }
    }

#else
    assert(false && "Eigen inverse not implemented for C++20");

#endif
#endif

    matvec_naive(in1_mds, in2_mds, out_mds);
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void matvec_to(auto &&in1, auto &&in2, auto &&out) {
    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::matvec_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 1, 1>{},
        std::integer_sequence<bool, true, true, false>{},
        std::forward<decltype(in1)>(in1), std::forward<decltype(in2)>(in2),
        std::forward<decltype(out)>(out));
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto matvec(auto &&in1, auto &&in2) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    const auto uin1_exts = core::slice_extents_from_right<2>(in1_mds.extents());
    const auto uin2_exts = core::slice_extents_from_right<2>(in2_mds.extents());
    const auto uout_exts = core::extents<
        core::common_index_type_t<typename decltype(uin1_exts)::index_type,
                                  typename decltype(uin2_exts)::index_type>,
        decltype(uin1_exts)::static_extent(0)>{uin1_exts.extent(0)};

    auto out = core::make_output<dtype>(std::index_sequence<2, 1>{}, uout_exts,
                                        in1_mds, in2_mds);

    matvec_to<backend>(in1_mds, in2_mds, out);

    return out;
}

} // namespace mdtensor::linalg
