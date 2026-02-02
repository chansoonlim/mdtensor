/**
 * @file
 * @brief Eigen interop utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#ifdef MDTENSOR_USE_EIGEN
#include <Eigen/Dense>
#endif

#include "../convert.hpp"
#include "../type.hpp"

#ifdef MDTENSOR_USE_EIGEN

namespace mdtensor {
namespace core {
namespace eigen {
namespace detail {

[[nodiscard]] inline constexpr int ext_to_dyn(size_t &&extent) noexcept {
    if (extent == dyn) {
        return Eigen::Dynamic;

    } else if (extent < std::numeric_limits<int>::max()) {
        return static_cast<int>(extent);

    } else {
        assert(false);
        return Eigen::Dynamic;
    }
}

} // namespace detail

template <typename T>
concept eigen_mappable_mdspan_c =
    (md_c<T> &&
     (std::is_same_v<typename std::remove_cvref_t<T>::layout_type,
                     layout_right> ||
      std::is_same_v<typename std::remove_cvref_t<T>::layout_type,
                     layout_left>) &&
     std::remove_cvref_t<T>::rank() == 2 &&
     std::remove_cvref_t<T>::is_always_unique() &&
     std::remove_cvref_t<T>::is_always_exhaustive() &&
     std::remove_cvref_t<T>::is_always_strided());

template <typename in_t>
[[nodiscard]] inline constexpr auto to_eigen(in_t &&in) noexcept {
    static_assert(eigen_mappable_mdspan_c<in_t>,
                  "Input mdspan is not eigen mappable");

    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    using Scalar = typename in_mds_t::element_type;
    using Lay = typename in_mds_t::layout_type;

    constexpr int RowsAtCompileTime =
        detail::ext_to_dyn(in_mds_t::static_extent(0));

    constexpr int ColsAtCompileTime =
        detail::ext_to_dyn(in_mds_t::static_extent(1));

    constexpr auto Options = []() {
        if constexpr (ColsAtCompileTime == 1 && RowsAtCompileTime != 1) {
            return Eigen::ColMajor;

        } else if constexpr (std::is_same_v<Lay, layout_right>) {
            return Eigen::RowMajor;

        } else if constexpr (std::is_same_v<Lay, layout_left>) {
            return Eigen::ColMajor;

        } else {
            static_assert(false, "Invalid layout type for Eigen mapping");
        }
    }();

    const int Rows = detail::ext_to_dyn(in_mds.extent(0));
    const int Cols = detail::ext_to_dyn(in_mds.extent(1));

    using eigen_t = std::conditional_t<
        std::is_const_v<Scalar>,
        const Eigen::Matrix<std::remove_const_t<Scalar>, RowsAtCompileTime,
                            ColsAtCompileTime, Options>,
        Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>>;

    return Eigen::Map<eigen_t>{in_mds.data_handle(), Rows, Cols};
}

} // namespace eigen
} // namespace core
} // namespace mdtensor

#endif // MDTENSOR_USE_EIGEN
