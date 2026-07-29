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

constexpr void matmul_to(auto &&...elems) {
    linalg::matmul_to(std::forward<decltype(elems)>(elems)...);
}

template <typename dtype = void>
[[nodiscard]] constexpr auto matmul(auto &&...elems) {
    return linalg::matmul<dtype>(std::forward<decltype(elems)>(elems)...);
}

constexpr void matvec_to(auto &&...elems) {
    linalg::matvec_to(std::forward<decltype(elems)>(elems)...);
}

template <typename dtype = void>
[[nodiscard]] constexpr auto matvec(auto &&...elems) {
    return linalg::matvec<dtype>(std::forward<decltype(elems)>(elems)...);
}

constexpr void vecmat_to(auto &&...elems) {
    linalg::vecmat_to(std::forward<decltype(elems)>(elems)...);
}

template <typename dtype = void>
[[nodiscard]] constexpr auto vecmat(auto &&...elems) {
    return linalg::vecmat<dtype>(std::forward<decltype(elems)>(elems)...);
}

} // namespace mdtensor
