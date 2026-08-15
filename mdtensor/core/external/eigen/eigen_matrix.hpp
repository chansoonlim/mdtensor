/**
 * @file
 * @brief Eigen matrix interop utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "eigen_extent.hpp"

namespace mdtensor::core::eigen {
namespace detail {

template <typename T> consteval bool is_eigen_matrix_mappable_impl() {
    using mds_t = decltype(to_mdspan(std::declval<T>()));

    using element_t = typename mds_t::element_type;
    using layout_t = typename mds_t::layout_type;
    using accessor_t = typename mds_t::accessor_type;

    constexpr bool supported_layout =
        std::same_as<layout_t, core::stdex::layout_right> ||
        std::same_as<layout_t, core::stdex::layout_left>;

    constexpr bool supported_accessor =
        std::same_as<accessor_t, core::stdex::default_accessor<element_t>>;

    return mds_t::rank() == 2 && supported_layout && supported_accessor &&
           mds_t::is_always_unique() && mds_t::is_always_exhaustive() &&
           mds_t::is_always_strided();
}

template <typename T>
struct is_eigen_matrix_mappable
    : std::bool_constant<is_eigen_matrix_mappable_impl<T>()> {};

template <typename T>
inline constexpr bool is_eigen_matrix_mappable_v =
    is_eigen_matrix_mappable<T>::value;

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

} // namespace detail

template <typename T>
concept eigen_matrix_mappable_c =
    detail::is_eigen_matrix_mappable_v<std::remove_cvref_t<T>>;

namespace detail {

template <typename in_t>
    requires eigen_matrix_mappable_c<in_t>
[[nodiscard]] auto make_eigen_matrix_view(in_t &&in) {
    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));
    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;

    static_assert(in_mds_t::rank() == 2,
                  "Input tensor must be rank-2 for Eigen matrix mapping.");

    constexpr int static_rows =
        to_eigen_static_extent<in_mds_t::static_extent(0)>();
    constexpr int static_cols =
        to_eigen_static_extent<in_mds_t::static_extent(1)>();
    constexpr auto option =
        detail::get_storage_option<typename in_mds_t::layout_type, static_rows,
                                   static_cols>();

    const Eigen::Index rows = to_eigen_extent(in_mds.extent(0));
    const Eigen::Index cols = to_eigen_extent(in_mds.extent(1));

    using matrix_t = Eigen::Matrix<typename in_mds_t::value_type, static_rows,
                                   static_cols, option>;

    using map_t = Eigen::Map<
        std::conditional_t<std::is_const_v<typename in_mds_t::element_type>,
                           const matrix_t, matrix_t>,
        Eigen::Unaligned>;

    if constexpr (static_rows == Eigen::Dynamic ||
                  static_cols == Eigen::Dynamic) {
        return map_t{in_mds.data_handle(), rows, cols};

    } else {
        return map_t{in_mds.data_handle()};
    }
}

[[nodiscard]] auto make_eigen_matrix_copy(auto &&in) {
    const auto in_mds = core::to_mdspan(std::forward<decltype(in)>(in));
    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;

    static_assert(in_mds_t::rank() == 2,
                  "Input tensor must be rank-2 for Eigen matrix copy.");

    constexpr int static_rows =
        to_eigen_static_extent<in_mds_t::static_extent(0)>();
    constexpr int static_cols =
        to_eigen_static_extent<in_mds_t::static_extent(1)>();
    constexpr auto option =
        detail::get_storage_option<typename in_mds_t::layout_type, static_rows,
                                   static_cols>();

    using matrix_t = Eigen::Matrix<typename in_mds_t::value_type, static_rows,
                                   static_cols, option>;

    matrix_t out{};

    if constexpr (static_rows == Eigen::Dynamic ||
                  static_cols == Eigen::Dynamic) {
        out.resize(to_eigen_extent(in_mds.extent(0)),
                   to_eigen_extent(in_mds.extent(1)));
    }

    using index_t = typename in_mds_t::index_type;

    for (index_t i = 0; i < in_mds.extent(0); i++) {
        for (index_t j = 0; j < in_mds.extent(1); j++) {
            out.coeffRef(i, j) = in_mds(i, j);
        }
    }

    return out;
}

} // namespace detail

template <CopyMode copy = CopyMode::AUTO>
[[nodiscard]] auto to_eigen_matrix(auto &&in) {
    const auto in_mds = core::to_mdspan(std::forward<decltype(in)>(in));
    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;

    static_assert(in_mds_t::rank() == 2,
                  "Input tensor must be rank-2 for Eigen matrix conversion.");

    constexpr bool can_borrow =
        std::is_lvalue_reference_v<decltype(in)> ||
        core::mdspan_c<std::remove_cvref_t<decltype(in)>>;

    constexpr bool can_map = eigen_matrix_mappable_c<decltype(in_mds)>;

    if constexpr (copy == CopyMode::AUTO) {
        if constexpr (can_borrow && can_map) {
            return detail::make_eigen_matrix_view(in_mds);

        } else {
            return detail::make_eigen_matrix_copy(in_mds);
        }

    } else if constexpr (copy == CopyMode::TRUE) {
        return detail::make_eigen_matrix_copy(in_mds);

    } else if constexpr (copy == CopyMode::FALSE) {
        static_assert(can_borrow,
                      "Eigen matrix conversion error: zero-copy mapping cannot "
                      "bind to a temporary owning tensor.");
        static_assert(
            can_map,
            "Eigen matrix conversion error: zero-copy mapping requires a "
            "rank-2, unique, strided mdspan with direct scalar storage.");

        return detail::make_eigen_matrix_view(in_mds);
    }
}

} // namespace mdtensor::core::eigen
