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

namespace mdtensor::core {
namespace detail {

template <std::size_t brank, mdspan_c io_t, mdspan_c... ios_t>
constexpr void batch_impl_native(auto &&func, io_t &&io, ios_t &&...ios) {
    if constexpr (brank == 0) {
        func(std::forward<io_t>(io), std::forward<ios_t>(ios)...);

    } else {
        using index_t = typename std::remove_cvref_t<io_t>::index_type;

        for (index_t i = 0; i < io.extent(0); i++) {
            batch_impl_native<brank - 1>(
                std::forward<decltype(func)>(func),
                submdspan_from_left(std::forward<io_t>(io), i),
                submdspan_from_left(std::forward<ios_t>(ios), i)...);
        }
    }
}

#ifdef MDTENSOR_USE_OPENMP

template <std::size_t brank, mdspan_c io_t, mdspan_c... ios_t>
void batch_impl_openmp(auto &&func, io_t &&io, ios_t &&...ios) {
    if constexpr (brank == 0) {
        func(std::forward<io_t>(io), std::forward<ios_t>(ios)...);

    } else {
        // Parallelize only the outermost batch axis.
        using index_t = typename std::remove_cvref_t<io_t>::index_type;

#pragma omp parallel for
        for (index_t i = 0; i < io.extent(0); i++) {
            batch_impl_native<brank - 1>(
                std::forward<decltype(func)>(func),
                submdspan_from_left(std::forward<io_t>(io), i),
                submdspan_from_left(std::forward<ios_t>(ios), i)...);
        }
    }
}

#endif

} // namespace detail

template <core::Backend backend, std::size_t brank>
constexpr void batch(auto &&func, auto &&...ios) {
    // TODO: assert when backend is not specified in each funciton call
    // assert(backend != core::Backend::AUTO);
    [[maybe_unused]] constexpr auto be = [&]() {
        if constexpr (backend == core::Backend::AUTO) {
            return core::Backend::NATIVE; // temporary approach.

        } else {
            return backend;
        }
    }();

    if constexpr (
#ifdef MDTENSOR_USE_OPENMP
        be == core::Backend::OPENMP
#else
        false
#endif
    ) {
#ifdef MDTENSOR_USE_OPENMP
        detail::batch_impl_openmp<brank>(
            std::forward<decltype(func)>(func),
            core::to_mdspan(std::forward<decltype(ios)>(ios))...);
#endif

    } else {
        detail::batch_impl_native<brank>(
            std::forward<decltype(func)>(func),
            core::to_mdspan(std::forward<decltype(ios)>(ios))...);
    }
}

template <core::Backend backend, std::size_t... uranks, bool... bcast>
constexpr void batch_with_broadcast(auto &&func, std::index_sequence<uranks...>,
                                    std::integer_sequence<bool, bcast...>,
                                    auto &&...ios) {
    // broadcast which bcast = true
    const auto [ios_bcast, bexts] =
        broadcast(std::index_sequence<uranks...>{},
                  std::integer_sequence<bool, bcast...>{},
                  std::forward<decltype(ios)>(ios)...);

    // batch
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        batch<backend, bexts.rank()>(std::forward<decltype(func)>(func),
                                     std::get<Is>(ios_bcast)...);
    }(std::make_index_sequence<sizeof...(ios)>{});
}

template <core::Backend backend, bool... bcast>
constexpr void batch_with_broadcast(auto &&func,
                                    std::integer_sequence<bool, bcast...>,
                                    auto &&...ios) {
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        batch_with_broadcast<backend>(std::forward<decltype(func)>(func),
                                      std::index_sequence<((void)Is, 0)...>{},
                                      std::integer_sequence<bool, bcast...>{},
                                      std::forward<decltype(ios)>(ios)...);
    }(std::make_index_sequence<sizeof...(ios)>{});
}

} // namespace mdtensor::core
