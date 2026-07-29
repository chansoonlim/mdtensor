/**
 * @file
 * @brief Eigen interop utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <Eigen/Dense>

#include "../mdspan.hpp"
#include "../type.hpp"

namespace mdtensor::core::eigen {
namespace detail {

template <std::size_t Extent>
[[nodiscard]] consteval int to_eigen_static_extent() {
    static_assert(
        Extent == core::dyn ||
            Extent <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "Static extent value exceeds maximum int value for Eigen mapping.");

    if constexpr (Extent == core::dyn) {
        return Eigen::Dynamic;

    } else {
        return static_cast<int>(Extent);
    }
}

template <core::integral_c ext_t>
[[nodiscard]] constexpr Eigen::Index to_eigen_extent(ext_t &&ext) {
    if constexpr (std::signed_integral<ext_t>) {
        if (ext < 0) {
            throw std::invalid_argument(
                "Negative extent value is invalid for Eigen mapping.");
        }
    }

    if (!std::in_range<Eigen::Index>(ext)) {
        throw std::invalid_argument(
            "Extent value exceeds maximum int value for Eigen mapping.");
    }

    return static_cast<Eigen::Index>(ext);
}

template <typename Layout, int Rows, int Cols>
consteval int get_storage_option() {
    if constexpr (Rows == 1 && Cols != 1) {
        return Eigen::RowMajor;

    } else if constexpr (Cols == 1 && Rows != 1) {
        return Eigen::ColMajor;

    } else if constexpr (std::same_as<Layout, core::stdex::layout_right>) {
        return Eigen::RowMajor;

    } else {
        return Eigen::ColMajor;
    }
}

template <typename T> consteval bool evaluate_eigen_mappable() {
    using mds_t = core::to_mdspan_t<T>;

    if constexpr (!mdspan_c<mds_t>) {
        return false;

    } else {
        return (std::same_as<typename mds_t::layout_type,
                             core::stdex::layout_right> ||
                std::same_as<typename mds_t::layout_type,
                             core::stdex::layout_left>) &&
               mds_t::rank() == 2 && mds_t::is_always_unique() &&
               mds_t::is_always_exhaustive() && mds_t::is_always_strided();
    }
}

template <typename T, typename = void>
struct is_eigen_mappable_impl : std::false_type {};

template <typename T>
struct is_eigen_mappable_impl<T, std::void_t<core::to_mdspan_t<T>>>
    : std::bool_constant<evaluate_eigen_mappable<T>()> {};

} // namespace detail

template <typename T>
struct is_eigen_mappable
    : detail::is_eigen_mappable_impl<std::remove_cvref_t<T>> {};

template <typename T>
inline constexpr bool is_eigen_mappable_v = is_eigen_mappable<T>::value;

template <typename T>
concept eigen_mappable_c = is_eigen_mappable_v<T>;

template <eigen_mappable_c in_t> [[nodiscard]] auto to_eigen(in_t &&in) {
    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));
    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;

    constexpr int static_rows =
        detail::to_eigen_static_extent<in_mds_t::static_extent(0)>();

    constexpr int static_cols =
        detail::to_eigen_static_extent<in_mds_t::static_extent(1)>();

    constexpr auto option =
        detail::get_storage_option<typename in_mds_t::layout_type, static_rows,
                                   static_cols>();

    const Eigen::Index rows = detail::to_eigen_extent(in_mds.extent(0));
    const Eigen::Index cols = detail::to_eigen_extent(in_mds.extent(1));

    using matrix_t = Eigen::Matrix<typename in_mds_t::value_type, static_rows,
                                   static_cols, option>;

    using mapped_matrix_t =
        std::conditional_t<std::is_const_v<typename in_mds_t::element_type>,
                           const matrix_t, matrix_t>;

    return Eigen::Map<mapped_matrix_t, Eigen::Unaligned>{in_mds.data_handle(),
                                                         rows, cols};
}

} // namespace mdtensor::core::eigen
