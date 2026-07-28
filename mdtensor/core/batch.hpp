/**
 * @file
 * @brief Batch utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "broadcast.hpp"
#include "container.hpp"
#include "extents.hpp"
#include "mdspan.hpp"
#include "submdspan.hpp"

namespace mdtensor {
namespace core {
namespace detail {

template <size_t brank, typename func_t, mdspan_c io_t, mdspan_c... ios_t>
inline constexpr void batch_impl_none(func_t &&func, io_t &&io,
                                      ios_t &&...ios) {
    using index_t = typename std::remove_cvref_t<io_t>::index_type;

    if constexpr (brank == 0) {
        std::forward<func_t>(func)(std::forward<io_t>(io),
                                   std::forward<ios_t>(ios)...);

    } else {
        for (index_t i = 0; i < io.extent(0); i++) {
            batch_impl_none<brank - 1>(
                std::forward<func_t>(func),
                submdspan_from_left(std::forward<io_t>(io), i),
                submdspan_from_left(std::forward<ios_t>(ios), i)...);
        }
    }
}

#if defined(_OPENMP) && defined(REAL_GCC)

template <size_t brank, typename func_t, mdspan_c io_t, mdspan_c... ios_t>
inline constexpr void batch_impl_cpump(func_t &&func, io_t &&io,
                                       ios_t &&...ios) {
    using index_t = typename std::remove_cvref_t<io_t>::index_type;

    if constexpr (brank == 0) {
        std::forward<func_t>(func)(std::forward<io_t>(io),
                                   std::forward<ios_t>(ios)...);

    } else {
#pragma omp parallel for
        for (index_t i = 0; i < io.extent(0); i++) {
            // NOTE: parallelization applied to the outermost loop (brank == 1)
            batch_impl_none<brank - 1>(
                std::forward<func_t>(func),
                submdspan_from_left(std::forward<io_t>(io), i),
                submdspan_from_left(std::forward<ios_t>(ios), i)...);
        }
    }
}

#endif

template <MPMode mpmode, size_t brank, typename func_t, mdspan_c... ios_t>
inline constexpr void batch_impl(func_t &&func, ios_t &&...ios) {
    if constexpr (mpmode == MPMode::CPUMP) {
        batch_impl_cpump<brank>(std::forward<func_t>(func),
                                std::forward<ios_t>(ios)...);

    } else {
        batch_impl_none<brank>(std::forward<func_t>(func),
                               std::forward<ios_t>(ios)...);
    }
}

} // namespace detail

template <typename dtype = void, size_t... uranks, extents_c uout_exts_t,
          typename... ins_t>
[[nodiscard]] inline constexpr auto create_out(std::index_sequence<uranks...>,
                                               uout_exts_t &&uout_exts,
                                               ins_t &&...ins) {
    static_assert(sizeof...(uranks) == sizeof...(ins_t),
                  "Number of uranks must match number of inputs.");

    constexpr size_t ins_num = sizeof...(uranks);

    if constexpr (ins_num == 0) {
        return make_container<dtype>(std::forward<uout_exts_t>(uout_exts));

    } else {
        using value_t =
            std::conditional_t<!std::is_void_v<dtype>, dtype,
                               common_data_type_t<value_type_t<ins_t>...>>;

        // calculate mdspans for inputs
        auto ins_mds =
            std::make_tuple(to_const_mdspan(std::forward<ins_t>(ins))...);

        // calculate broadcasted extents
        const auto bexts = detail::get_broadcast_extents(
            std::index_sequence<uranks...>{}, ins_mds);

        // make output container
        return make_container<value_t>(
            compose_extents(bexts, std::forward<uout_exts_t>(uout_exts)));
    }
}

template <typename dtype = void, extents_c uout_exts_t, typename... ins_t>
[[nodiscard]] inline constexpr auto create_out(uout_exts_t &&uout_exts,
                                               ins_t &&...ins) {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return create_out<dtype>(std::index_sequence<((void)Is, 0)...>{},
                                 std::forward<uout_exts_t>(uout_exts),
                                 std::forward<ins_t>(ins)...);
    }(std::make_index_sequence<sizeof...(ins_t)>{});
}

template <typename dtype = void, size_t... uranks,
          extents_tuple_c uout_exts_tuple_t, typename... ins_t>
[[nodiscard]] inline constexpr auto
create_outs(std::index_sequence<uranks...>, uout_exts_tuple_t &&uout_exts_tuple,
            ins_t &&...ins) {
    static_assert(sizeof...(uranks) == sizeof...(ins_t),
                  "Number of uranks must match number of inputs.");

    constexpr size_t ins_num = sizeof...(uranks);
    constexpr size_t outs_num =
        std::tuple_size_v<std::remove_cvref_t<uout_exts_tuple_t>>;

    if constexpr (ins_num == 0) {
        return [&]<size_t... Is>(std::index_sequence<Is...>) {
            return std::tuple{
                make_container<dtype>(std::get<Is>(uout_exts_tuple))...};
        }(std::make_index_sequence<outs_num>{});

    } else {
        using value_t =
            std::conditional_t<!std::is_void_v<dtype>, dtype,
                               common_data_type_t<value_type_t<ins_t>...>>;

        // calculate mdspans for inputs
        auto ins_mds =
            std::make_tuple(to_const_mdspan(std::forward<ins_t>(ins))...);

        // calculate broadcasted extents
        const auto bexts = detail::get_broadcast_extents(
            std::index_sequence<uranks...>{}, ins_mds);

        // make output container
        return [&]<size_t... Is>(std::index_sequence<Is...>) {
            return std::tuple{make_container<value_t>(
                compose_extents(bexts, std::get<Is>(uout_exts_tuple)))...};
        }(std::make_index_sequence<outs_num>{});
    }
}

template <typename dtype = void, extents_tuple_c uout_exts_tuple_t,
          typename... ins_t>
[[nodiscard]] inline constexpr auto
create_outs(uout_exts_tuple_t &&uout_exts_tuple, ins_t &&...ins) {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return create_outs<dtype>(
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<uout_exts_tuple_t>(uout_exts_tuple),
            std::forward<ins_t>(ins)...);
    }(std::make_index_sequence<sizeof...(ins_t)>{});
}

template <MPMode mpmode, typename func_t, size_t... uranks, bool... writable,
          typename... ios_t>
inline constexpr void batch(func_t &&func, std::index_sequence<uranks...>,
                            std::integer_sequence<bool, writable...>,
                            ios_t &&...ios) {
    const auto [ios_bcast, bexts] =
        broadcast(std::index_sequence<uranks...>{},
                  std::integer_sequence<bool, writable...>{},
                  std::forward<ios_t>(ios)...);

    [&]<size_t... Is>(std::index_sequence<Is...>) {
        detail::batch_impl<mpmode, bexts.rank()>(std::forward<func_t>(func),
                                                 std::get<Is>(ios_bcast)...);
    }(std::make_index_sequence<sizeof...(ios_t)>{});
}

template <MPMode mpmode, typename func_t, bool... writable, typename... ios_t>
inline constexpr void
batch(func_t &&func, std::integer_sequence<bool, writable...>, ios_t &&...ios) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        batch<mpmode>(std::forward<func_t>(func),
                      std::index_sequence<((void)Is, 0)...>{},
                      std::integer_sequence<bool, writable...>{},
                      std::get<Is>(std::forward_as_tuple(
                          std::forward<ios_t>(ios)...))...);
    }(std::make_index_sequence<sizeof...(ios_t)>{});
}

} // namespace core
} // namespace mdtensor
