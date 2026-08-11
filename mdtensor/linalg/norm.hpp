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

namespace mdtensor::linalg {
namespace ufunc {

constexpr void norm_ufunc(auto &&in, auto &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using index_t = typename decltype(in_mds)::index_type;

    out_mds() = 0;
    for (index_t i = 0; i < in_mds.extent(0); i++) {
        out_mds() += in_mds(i) * in_mds(i);
    }

    if (out_mds() > 0) {
        out_mds() = sqrt(out_mds());
    }
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void norm_to(auto &&in, auto &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    if constexpr (backend == core::Backend::SIMD) {
        static_cast<void>(sum<-1, void, false, backend>(
            multiply<void, backend>(in_mds, in_mds), out_mds));
        static_cast<void>(sqrt<void, backend>(out_mds, out_mds));

    } else {
        core::batch_with_broadcast<backend>(
            [](auto &&...elems) {
                ufunc::norm_ufunc(std::forward<decltype(elems)>(elems)...);
            },
            std::index_sequence<1, 0>{},
            std::integer_sequence<bool, true, false>{}, in_mds, out_mds);
    }
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto norm(auto &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out = core::make_broadcasted_tensor<dtype>(
        std::index_sequence<1>{}, core::extents<std::uint8_t>{}, in_mds);

    norm_to<backend>(in_mds, out);

    return out;
}

} // namespace mdtensor::linalg
