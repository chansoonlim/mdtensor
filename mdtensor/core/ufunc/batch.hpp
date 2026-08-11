/**
 * @file
 * @brief Batch utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../broadcast/broadcast.hpp"

namespace mdtensor::core {

enum class Backend {
    AUTO,   // Automatically select backend based on input types and sizes
    NATIVE, // Native mdtensor implementation
    SIMD,   // SIMD parallelization

#ifdef MDTENSOR_USE_EIGEN
    EIGEN, // Eigen backend
#endif

#ifdef MDTENSOR_USE_OPENMP
    OPENMP, // CPU multi-processing with OpenMP
#endif
};

namespace detail {

template <std::size_t brank, bool has_escape, mdspan_c io_t, mdspan_c... ios_t>
constexpr decltype(auto) batch_impl_native(auto &&ufunc, io_t &&io,
                                           ios_t &&...ios) {
    if constexpr (brank == 0) {
        if constexpr (has_escape) {
            return ufunc(unwrap_scalar(std::forward<io_t>(io)),
                         unwrap_scalar(std::forward<ios_t>(ios))...);

        } else {
            ufunc(unwrap_scalar(std::forward<io_t>(io)),
                  unwrap_scalar(std::forward<ios_t>(ios))...);
            return;
        }

    } else {
        using index_t = typename std::remove_cvref_t<io_t>::index_type;

        if constexpr (has_escape) {
            for (index_t i = 0; i < io.extent(0); i++) {
                if (!batch_impl_native<brank - 1, has_escape>(
                        std::forward<decltype(ufunc)>(ufunc),
                        submdspan_from_left(std::forward<io_t>(io), i),
                        submdspan_from_left(std::forward<ios_t>(ios), i)...)) {
                    return false;
                }
            }
            return true;

        } else {
            for (index_t i = 0; i < io.extent(0); i++) {
                batch_impl_native<brank - 1, has_escape>(
                    std::forward<decltype(ufunc)>(ufunc),
                    submdspan_from_left(std::forward<io_t>(io), i),
                    submdspan_from_left(std::forward<ios_t>(ios), i)...);
            }
            return;
        }
    }
}

#ifdef MDTENSOR_USE_OPENMP

template <std::size_t brank, mdspan_c io_t, mdspan_c... ios_t>
constexpr void batch_impl_openmp(auto &&ufunc, io_t &&io, ios_t &&...ios) {
    // NOTE: OpenMP does not support return value from parallel region.
    // NOTE: Parallelize only the outermost batch axis.

    if constexpr (brank == 0) {
        ufunc(unwrap_scalar(std::forward<io_t>(io)),
              unwrap_scalar(std::forward<ios_t>(ios))...);

    } else {
        using index_t = typename std::remove_cvref_t<io_t>::index_type;

#pragma omp parallel for
        for (index_t i = 0; i < io.extent(0); i++) {
            batch_impl_native<brank - 1, false>(
                std::forward<decltype(ufunc)>(ufunc),
                submdspan_from_left(std::forward<io_t>(io), i),
                submdspan_from_left(std::forward<ios_t>(ios), i)...);
        }
    }
}

#endif

} // namespace detail

template <Backend backend, std::size_t brank, bool has_escape = false>
constexpr decltype(auto) batch(auto &&ufunc, auto &&...ios) {
#ifdef MDTENSOR_USE_OPENMP
    // TODO: assert when backend is not specified in each funciton call
    // assert(backend != Backend::AUTO);
    constexpr auto backend_real = [&]() {
        if constexpr (backend == Backend::AUTO) {
            return Backend::NATIVE; // temporary approach.

        } else {
            return backend;
        }
    }();
#endif

    if constexpr (
#ifdef MDTENSOR_USE_OPENMP
        backend_real == Backend::OPENMP
#else
        false
#endif
    ) {
#ifdef MDTENSOR_USE_OPENMP
        static_assert(!has_escape,
                      "OpenMP backend does not support return value.");
        detail::batch_impl_openmp<brank>(
            std::forward<decltype(ufunc)>(ufunc),
            to_mdspan(std::forward<decltype(ios)>(ios))...);
        return;
#endif

    } else {
        return detail::batch_impl_native<brank, has_escape>(
            std::forward<decltype(ufunc)>(ufunc),
            to_mdspan(std::forward<decltype(ios)>(ios))...);
    }
}

template <Backend backend, bool has_escape = false, std::size_t... uranks,
          bool... bcast>
constexpr decltype(auto)
batch_with_broadcast(auto &&ufunc, std::index_sequence<uranks...>,
                     std::integer_sequence<bool, bcast...>, auto &&...ios) {
    // broadcast which bcast = true
    const auto [ios_bcast, bexts] =
        broadcast(std::index_sequence<uranks...>{},
                  std::integer_sequence<bool, bcast...>{},
                  std::forward<decltype(ios)>(ios)...);

    // batch
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return batch<backend, bexts.rank(), has_escape>(
            std::forward<decltype(ufunc)>(ufunc), std::get<Is>(ios_bcast)...);
    }(std::make_index_sequence<sizeof...(ios)>{});
}

template <Backend backend, bool has_escape = false, bool... bcast>
constexpr decltype(auto)
batch_with_broadcast(auto &&ufunc, std::integer_sequence<bool, bcast...>,
                     auto &&...ios) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return batch_with_broadcast<backend, has_escape>(
            std::forward<decltype(ufunc)>(ufunc),
            std::index_sequence<((void)Is, 0)...>{},
            std::integer_sequence<bool, bcast...>{},
            std::forward<decltype(ios)>(ios)...);
    }(std::make_index_sequence<sizeof...(ios)>{});
}

} // namespace mdtensor::core
