/**
 * @file
 * @brief Element-wise absolute value utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../creation/empty_like.hpp"

namespace mdtensor {
namespace ufunc {

constexpr void absolute_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    if constexpr (std::is_signed_v<std::remove_cvref_t<decltype(in())>>) {
#ifdef REAL_GCC // NOTE: std::abs is not constexpr in clang 16.
        out() = std::abs(in());
#else
        out() = in() < 0 ? -in() : in();
#endif

    } else {
        out() = in();
    }
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto absolute(auto &&in,
                                      out_t &&out = out_t{std::nullopt},
                                      where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::is_nullopt_t_c<decltype(out)>) {
            return empty_like<dtype>(in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::absolute_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{},
        std::forward<decltype(in)>(in), out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
