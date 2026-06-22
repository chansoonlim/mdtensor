/**
 * @file
 * @brief Linspace utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace detail {

template <md_c start_t, md_c stop_t, md_c out_t>
    requires(std::remove_cvref_t<start_t>::rank() == 0 &&
             std::remove_cvref_t<stop_t>::rank() == 0 &&
             std::remove_cvref_t<out_t>::rank() == 1)
inline constexpr void linspace_impl(start_t &&start, stop_t &&stop, out_t &&out,
                                    const bool endpoint = true) noexcept {
    using out_base_t = std::remove_cvref_t<out_t>;

    using value_t = typename out_base_t::value_type;
    using index_t = typename out_base_t::index_type;

    const index_t num = out.extent(0);

    if (num == 0) [[unlikely]] {
        // do nothing

    } else if (num == 1) [[unlikely]] {
        out(0) = static_cast<value_t>(start());

    } else {
        const value_t start_val = static_cast<value_t>(start());
        const value_t stop_val = static_cast<value_t>(stop());
        const value_t step =
            (stop_val - start_val) / (endpoint ? num - 1 : num);

        for (index_t i = 0; i < num; i++) {
            out(i) = start_val + step * i;
        }
    }
}

} // namespace detail

/**
 * @brief Generate evenly spaced samples along a specified axis (in-place).
 *
 * @tparam Axis (optional) Axis in the output tensor along which samples are
 * generated. Default is 0. Negative values are supported (Numpy-style).
 * @tparam start_t Start values (mdspan, mdarray, scalar, etc.).
 * @tparam stop_t  Stop values (mdspan, mdarray, scalar, etc.).
 * @tparam out_t   Output buffer (mdspan, mdarray, etc.).
 *
 * @param start Start value(s). Must have the same rank as `stop`.
 * @param stop Stop value(s). Must have the same rank as `start`.
 * @param out Output buffer whose rank must be `rank(start) + 1`.
 * @param endpoint If true, include `stop` as the last sample along the linspace
 * axis. If false, exclude `stop`.
 *
 * @details
 * - If `start` and `stop` are scalars (rank 0), `out` must be rank 1 and this
 *   fills `out` directly.
 * - If `start` and `stop` are higher-rank, this function recursively applies
 *   linspace along `Axis` without broadcasting.
 *
 * @note This implementation currently **does not** support broadcasting between
 * `start` and `stop`. Rank must match and `out` must be exactly one higher
 * rank.
 * @note The axis selection is normalized to `[0, out_rank)` using modulo rules.
 *
 * @warning The function uses runtime `assert` checks for certain extent
 * invariants when rank > 0.
 *
 * @see mdtensor::linspace for the out-of-place version that allocates and
 * returns the output.
 */
template <int64_t Axis = 0, typename start_t, typename stop_t, typename out_t>
inline constexpr void linspace_to(start_t &&start, stop_t &&stop, out_t &&out,
                                  const bool endpoint = true) noexcept {
    const auto start_mds = core::to_const_mdspan(std::forward<start_t>(start));
    const auto stop_mds = core::to_const_mdspan(std::forward<stop_t>(stop));
    const auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    using start_mds_t = decltype(start_mds);
    using stop_mds_t = decltype(stop_mds);
    using out_mds_t = decltype(out_mds);

    static_assert(start_mds_t::rank() == stop_mds_t::rank() &&
                      start_mds_t::rank() + 1 == out_mds_t::rank(),
                  "linspace does not support broadcasting");
    // TODO: support broadcasting without changing rank.

    if constexpr (start_mds_t::rank() == 0) {
        detail::linspace_impl(start_mds, stop_mds, out_mds, endpoint);

    } else {
        constexpr size_t out_rank = out_mds_t::rank();
        constexpr size_t axis = static_cast<size_t>(
            ((Axis % static_cast<int64_t>(out_rank)) + (out_rank)) % out_rank);
        constexpr size_t lspace = axis == 0 ? 1 : 0;

        assert(start_mds.extent(0) == stop_mds.extent(0));
        assert(start_mds.extent(0) == out_mds.extent(lspace));

        for (typename out_mds_t::index_type i = 0; i < out_mds.extent(lspace);
             i++) {
            linspace_to<axis - (lspace == 0 ? 1 : 0)>(
                core::submdspan_from_left(start_mds, i),
                core::submdspan_from_left(stop_mds, i),
                core::submdspan_from_left<lspace>(out_mds, i), endpoint);
        }
    }
}

/**
 * @brief Create an array of evenly spaced samples over an interval
 * (out-of-place).
 *
 * @tparam Axis (optional) Axis of the output along which samples are placed.
 * Default is 0. Negative values are supported (Python-style).
 * @tparam exts_t (optional) 1D extents type that determines the number of
 * samples. Default is `extents<uint8_t, 50>` (i.e., static 50 samples).
 * @tparam dtype (optional) Output value type. If `void`, deduced as a common
 * type of start/stop value types.
 * @tparam start_t Start values (mdspan, mdarray, scalar, etc.).
 * @tparam stop_t  Stop values (mdspan, mdarray, scalar, etc.).
 *
 * @param start Start value(s).
 * @param stop Stop value(s).
 * @param exts  1D extents specifying the number of samples (length `num`).
 * @param endpoint If true, include `stop` as the last sample; otherwise
 * exclude.
 *
 * @return An mdarray whose rank is `rank(start) + 1`. The output shape is
 * formed by inserting the 1D `exts` at the specified `Axis` into the base
 * extents of `start`:
 * - `out_extents = concat(slice_left(bexts, axis), exts, slice_right(bexts,
 * ...))`
 *
 * @note Broadcasting between `start` and `stop` is currently not supported.
 * They must have the same rank and compatible extents.
 *
 * @see mdtensor::linspace_to for the in-place version that writes into an
 * existing output.
 */
template <int64_t Axis = 0, extents_c exts_t = extents<uint8_t, 50>,
          typename dtype = void, typename start_t, typename stop_t>
    requires(exts_t::rank() == 1)
[[nodiscard]] inline constexpr auto
linspace(start_t &&start, stop_t &&stop, const exts_t &exts = exts_t{},
         const bool endpoint = true) noexcept {
    const auto start_mds = core::to_const_mdspan(std::forward<start_t>(start));
    const auto stop_mds = core::to_const_mdspan(std::forward<stop_t>(stop));

    using start_mds_t = decltype(start_mds);
    using stop_mds_t = decltype(stop_mds);

    using value_t = std::conditional_t<
        !std::is_void_v<dtype>, dtype,
        core::common_data_type_t<typename start_mds_t::value_type,
                                 typename stop_mds_t::value_type>>;

    constexpr size_t out_rank = start_mds_t::rank() + 1;
    constexpr size_t axis = static_cast<size_t>(
        ((Axis % static_cast<int64_t>(out_rank)) + (out_rank)) % out_rank);

    const auto bexts = start_mds.extents();
    auto out = core::create_data<value_t>(core::concatenate(
        core::slice_from_left<axis>(bexts), exts,
        core::slice_from_right<decltype(bexts)::rank() - axis>(bexts)));

    linspace_to<Axis>(start_mds, stop_mds, out, endpoint);

    return out;
}

} // namespace mdtensor
