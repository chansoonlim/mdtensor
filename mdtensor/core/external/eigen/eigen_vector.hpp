/**
 * @file
 * @brief Eigen vector interop utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "eigen_extent.hpp"

namespace mdtensor::core::eigen {
namespace detail {

template <typename T>
[[nodiscard]] consteval bool is_eigen_vector_mappable_impl() {
    using mds_t = decltype(to_mdspan(std::declval<T>()));

    using element_t = typename mds_t::element_type;
    using layout_t = typename mds_t::layout_type;
    using accessor_t = typename mds_t::accessor_type;

    constexpr bool supported_layout =
        std::same_as<layout_t, core::stdex::layout_right> ||
        std::same_as<layout_t, core::stdex::layout_left> ||
        std::same_as<layout_t, core::stdex::layout_stride>;

    constexpr bool supported_accessor =
        std::same_as<accessor_t, core::stdex::default_accessor<element_t>>;

    return mds_t::rank() == 1 && supported_layout && supported_accessor &&
           !std::is_volatile_v<element_t> && mds_t::is_always_unique() &&
           mds_t::is_always_strided() &&
           std::convertible_to<typename mds_t::data_handle_type, element_t *>;
}

template <typename T>
struct is_eigen_vector_mappable
    : std::bool_constant<is_eigen_vector_mappable_impl<T>()> {};

template <typename T>
inline constexpr bool is_eigen_vector_mappable_v =
    is_eigen_vector_mappable<T>::value;

} // namespace detail

template <typename T>
concept eigen_vector_mappable_c =
    detail::is_eigen_vector_mappable_v<std::remove_cvref_t<T>>;

namespace detail {

template <bool col = true, typename in_t>
    requires eigen_vector_mappable_c<in_t>
[[nodiscard]] auto make_eigen_vector_view(in_t &&in) {
    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));
    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;

    static_assert(in_mds_t::rank() == 1,
                  "Input tensor must be rank-1 for Eigen vector mapping.");

    constexpr int static_size =
        to_eigen_static_extent<in_mds_t::static_extent(0)>();
    constexpr int static_rows = col ? static_size : 1;
    constexpr int static_cols = col ? 1 : static_size;
    constexpr auto option = col ? Eigen::ColMajor : Eigen::RowMajor;

    using vector_t = Eigen::Matrix<typename in_mds_t::value_type, static_rows,
                                   static_cols, option>;

    using stride_t = Eigen::InnerStride<Eigen::Dynamic>;

    using map_t = Eigen::Map<
        std::conditional_t<std::is_const_v<typename in_mds_t::element_type>,
                           const vector_t, vector_t>,
        Eigen::Unaligned, stride_t>;

    const stride_t stride{to_eigen_extent(in_mds.stride(0))};

    if constexpr (static_size == Eigen::Dynamic) {
        return map_t{in_mds.data_handle(), to_eigen_extent(in_mds.extent(0)),
                     stride};

    } else {
        return map_t{in_mds.data_handle(), stride};
    }
}

template <bool col = true>
[[nodiscard]] auto make_eigen_vector_copy(auto &&in) {
    const auto in_mds = core::to_mdspan(std::forward<decltype(in)>(in));
    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;

    static_assert(in_mds_t::rank() == 1,
                  "Input tensor must be rank-1 for Eigen vector copy.");

    constexpr int static_size =
        to_eigen_static_extent<in_mds_t::static_extent(0)>();
    constexpr int static_rows = col ? static_size : 1;
    constexpr int static_cols = col ? 1 : static_size;
    constexpr auto option = col ? Eigen::ColMajor : Eigen::RowMajor;

    using vector_t = Eigen::Matrix<typename in_mds_t::value_type, static_rows,
                                   static_cols, option>;

    auto out = [&]() {
        if constexpr (static_size == Eigen::Dynamic) {
            return vector_t{to_eigen_extent(in_mds.extent(0))};

        } else {
            return vector_t{};
        }
    }();

    using index_t = typename in_mds_t::index_type;

    for (index_t i = 0; i < in_mds.extent(0); i++) {
        out.coeffRef(i) = in_mds(i);
    }

    return out;
}

} // namespace detail

template <bool col = true, CopyMode copy = CopyMode::AUTO>
[[nodiscard]] auto to_eigen_vector(auto &&in) {
    const auto in_mds = core::to_mdspan(std::forward<decltype(in)>(in));
    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;

    static_assert(in_mds_t::rank() == 1,
                  "Input tensor must be rank-1 for Eigen vector conversion.");

    constexpr bool can_borrow =
        std::is_lvalue_reference_v<decltype(in)> ||
        core::mdspan_c<std::remove_cvref_t<decltype(in)>>;

    constexpr bool can_map = eigen_vector_mappable_c<decltype(in_mds)>;

    if constexpr (copy == CopyMode::AUTO) {
        if constexpr (can_borrow && can_map) {
            return detail::make_eigen_vector_view<col>(in_mds);

        } else {
            return detail::make_eigen_vector_copy<col>(in_mds);
        }

    } else if constexpr (copy == CopyMode::TRUE) {
        return detail::make_eigen_vector_copy<col>(in_mds);

    } else if constexpr (copy == CopyMode::FALSE) {
        static_assert(can_borrow,
                      "Eigen vector conversion error: zero-copy mapping cannot "
                      "bind to a temporary owning tensor.");
        static_assert(
            can_map,
            "Eigen vector conversion error: zero-copy mapping requires a "
            "rank-1, unique, strided mdspan with direct scalar storage.");

        return detail::make_eigen_vector_view<col>(in_mds);
    }
}

} // namespace mdtensor::core::eigen
