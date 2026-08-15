/**
 * @file
 * @brief Linear algebra module includes for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "cholesky.hpp"
#include "inv.hpp"
#include "lu.hpp"
#include "matmul.hpp"
#include "matvec.hpp"
#include "norm.hpp"
#include "solve.hpp"
#include "vecmat.hpp"

namespace mdtensor {

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto matmul(auto &&in1, auto &&in2,
                                    out_t &&out = out_t{std::nullopt}) {
    return linalg::matmul<dtype, backend>(std::forward<decltype(in1)>(in1),
                                          std::forward<decltype(in2)>(in2),
                                          std::forward<decltype(out)>(out));
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto matvec(auto &&in1, auto &&in2,
                                    out_t &&out = out_t{std::nullopt}) {
    return linalg::matvec<dtype, backend>(std::forward<decltype(in1)>(in1),
                                          std::forward<decltype(in2)>(in2),
                                          std::forward<decltype(out)>(out));
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto vecmat(auto &&in1, auto &&in2,
                                    out_t &&out = out_t{std::nullopt}) {
    return linalg::vecmat<dtype, backend>(std::forward<decltype(in1)>(in1),
                                          std::forward<decltype(in2)>(in2),
                                          std::forward<decltype(out)>(out));
}

} // namespace mdtensor
