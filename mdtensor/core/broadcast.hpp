/**
 * @file
 * @brief Broadcasting utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <optional>
#include <tuple>

#include "submdspan.hpp"
#include "type.hpp"

namespace mdtensor {
namespace core {

template <size_t offset, size_t rank, extents_c in_t>
[[nodiscard]] inline constexpr auto
slice_with_offset(in_t &&in = in_t{}) noexcept {
    using in_base_t = std::remove_cvref_t<in_t>;

    static_assert(in_base_t::rank() >= offset + rank,
                  "Incompatible offset and rank for slicing.");

    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return extents<typename in_base_t::index_type,
                       in_base_t::static_extent(offset + Is)...>{
            in.extent(offset + Is)...};
    }(std::make_index_sequence<rank>{});
}

template <size_t rank, extents_c in_t>
[[nodiscard]] inline constexpr auto
slice_from_left(in_t &&in = in_t{}) noexcept {
    return slice_with_offset<0, rank>(std::forward<in_t>(in));
}

template <size_t rank, extents_c in_t>
[[nodiscard]] inline constexpr auto
slice_from_right(in_t &&in = in_t{}) noexcept {
    using in_base_t = std::remove_cvref_t<in_t>;

    return slice_with_offset<in_base_t::rank() - rank, rank>(
        std::forward<in_t>(in));
}

template <extents_c in1_t, extents_c in2_t, extents_c... ins_t>
[[nodiscard]] inline constexpr auto concatenate(in1_t &&in1 = in1_t{},
                                                in2_t &&in2 = in2_t{},
                                                ins_t &&...ins) noexcept {
    using in1_base_t = std::remove_cvref_t<in1_t>;
    using in2_base_t = std::remove_cvref_t<in2_t>;
    using index_t = std::common_type_t<typename in1_base_t::index_type,
                                       typename in2_base_t::index_type>;

    const auto cexts =
        [&]<size_t... Is, size_t... Js>(std::index_sequence<Is...>,
                                        std::index_sequence<Js...>) {
            return extents<index_t, in1_base_t::static_extent(Is)...,
                           in2_base_t::static_extent(Js)...>{
                static_cast<index_t>(in1.extent(Is))...,
                static_cast<index_t>(in2.extent(Js))...};
        }(std::make_index_sequence<in1_base_t::rank()>{},
          std::make_index_sequence<in2_base_t::rank()>{});

    if constexpr (sizeof...(ins_t) == 0) {
        return cexts;

    } else {
        return concatenate(cexts, std::forward<ins_t>(ins)...);
    }
}

namespace detail {

template <size_t Is, size_t brank, extents_c in1_t, extents_c in2_t>
constexpr size_t broadcast_static_extent(in1_t &&in = in1_t{},
                                         in2_t &&in2 = in2_t{}) noexcept {
    constexpr size_t e1 =
        (Is < brank - in1_t::rank()
             ? 1
             : in1_t::static_extent(Is - (brank - in1_t::rank())));
    constexpr size_t e2 =
        (Is < brank - in2_t::rank()
             ? 1
             : in2_t::static_extent(Is - (brank - in2_t::rank())));
    static_assert(e1 == e2 || e1 == 1 || e1 == dyn || e2 == 1 || e2 == dyn,
                  "incompatible extents for broadcasting.");

    return std::max(e1, e2);
}

} // namespace detail

template <extents_c in_t>
[[nodiscard]] inline constexpr auto broadcast(in_t &&in = in_t{}) noexcept {
    return std::forward<in_t>(in);
}

template <extents_c in1_t, extents_c in2_t, extents_c... ins_t>
[[nodiscard]] inline constexpr auto broadcast(in1_t &&in1 = in1_t{},
                                              in2_t &&in2 = in2_t{},
                                              ins_t &&...ins) noexcept {
    using in1_base_t = std::remove_cvref_t<in1_t>;
    using in2_base_t = std::remove_cvref_t<in2_t>;
    using index_t = std::common_type_t<typename in1_base_t::index_type,
                                       typename in2_base_t::index_type>;

    constexpr size_t brank = std::max(in1_base_t::rank(), in2_base_t::rank());

    const auto bexts = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return extents<index_t, detail::broadcast_static_extent<
                                    Is, brank, in1_base_t, in2_base_t>()...>{
            [&]() {
                const size_t x1 =
                    (Is < brank - in1_base_t::rank()
                         ? 1
                         : in1.extent(Is - (brank - in1_base_t::rank())));
                const size_t x2 =
                    (Is < brank - in2_base_t::rank()
                         ? 1
                         : in2.extent(Is - (brank - in2_base_t::rank())));

                assert(x1 == x2 || x1 == 1 || x2 == 1);

                return std::max(x1, x2);
            }()...};
    }(std::make_index_sequence<brank>{});

    if constexpr (sizeof...(ins_t) == 0) {
        return bexts;

    } else {
        return broadcast(bexts, std::forward<ins_t>(ins)...);
    }
}

template <size_t offset, size_t brank, mdspan_c in_t, extents_c new_bexts_t>
[[nodiscard]] inline constexpr auto
broadcast_to(in_t &&in = in_t{},
             new_bexts_t &&new_bexts = new_bexts_t{}) noexcept {
    using in_base_t = std::remove_cvref_t<in_t>;
    using new_bexts_base_t = std::remove_cvref_t<new_bexts_t>;

    constexpr size_t urank = in_base_t::rank() - brank;

    static_assert(in_base_t::rank() >= offset + brank,
                  "Incompatible offset and brank for broadcasting.");
    static_assert(new_bexts_base_t::rank() >= brank,
                  "Incompatible brank for broadcasting.");

    if constexpr (brank == new_bexts_base_t::rank() &&
                  []<size_t... Is>(std::index_sequence<Is...>) {
                      return ((in_base_t::static_extent(offset + Is) != dyn &&
                               in_base_t::static_extent(offset + Is) ==
                                   new_bexts_base_t::static_extent(Is)) &&
                              ...);
                  }(std::make_index_sequence<brank>{})) {
        return std::forward<in_t>(in);

    } else if constexpr (in_base_t::rank() == 0) {
        using index_type = typename new_bexts_base_t::index_type;

        auto new_strides = std::array<index_type, new_bexts_base_t::rank()>{};

        for (size_t i = 0; i < new_bexts_base_t::rank(); i++) {
            new_strides[i] = 0;
        }

        return mdspan<typename in_base_t::element_type, new_bexts_base_t,
                      layout_stride, typename in_base_t::accessor_type>{
            in.data_handle(),
            layout_stride::mapping{std::forward<new_bexts_t>(new_bexts),
                                   new_strides}};

    } else {
        using index_type =
            std::common_type_t<typename in_base_t::index_type,
                               typename new_bexts_base_t::index_type>;

        auto new_strides =
            std::array<index_type, urank + new_bexts_base_t::rank()>{};

        for (size_t i = 0; i < new_strides.size(); i++) {
            if (i < offset) {
                new_strides[i] = static_cast<index_type>(in.stride(i));

            } else if (i < offset + new_bexts_base_t::rank() - brank) {
                new_strides[i] = 0;

            } else if (i < offset + new_bexts_base_t::rank()) {
                const size_t j = i + brank - new_bexts_base_t::rank();

                if (static_cast<size_t>(in.extent(j)) ==
                    static_cast<size_t>(new_bexts.extent(i - offset))) {
                    new_strides[i] = static_cast<index_type>(in.stride(j));

                } else if (in.extent(j) == 1) {
                    new_strides[i] = 0;

                } else {
                    assert(false);
                }

            } else {
                new_strides[i] = static_cast<index_type>(
                    in.stride(i + brank - new_bexts_base_t::rank()));
            }
        }

        const auto new_extents = [&]() {
            if constexpr (offset == 0) {
                return concatenate(std::forward<new_bexts_t>(new_bexts),
                                   slice_from_right<urank>(in.extents()));

            } else {
                return concatenate(
                    slice_from_left<offset>(in.extents()),
                    std::forward<new_bexts_t>(new_bexts),
                    slice_from_right<urank - offset>(in.extents()));
            }
        }();

        return mdspan<typename in_base_t::element_type,
                      std::remove_cvref_t<decltype(new_extents)>, layout_stride,
                      typename in_base_t::accessor_type>{
            in.data_handle(), layout_stride::mapping{new_extents, new_strides}};
    }
}

namespace detail {

template <size_t brank, typename func_t, size_t offset, size_t... offsets,
          mdspan_c in_t, mdspan_c... ins_t>
inline constexpr void batch_impl_none(func_t &&func,
                                      std::index_sequence<offset, offsets...>,
                                      in_t &&in, ins_t &&...ins) {
    static_assert(sizeof...(offsets) == sizeof...(ins),
                  "Number of offsets must match number of inputs.");

    using index_t = typename std::remove_cvref_t<in_t>::index_type;

    if constexpr (brank == 0) {
        std::forward<func_t>(func)(in, ins...);

    } else {
        for (index_t i = 0; i < in.extent(offset); i++) {
            batch_impl_none<brank - 1>(
                std::forward<func_t>(func),
                std::index_sequence<offset, offsets...>{},
                submdspan_from_left<offset>(in, i),
                submdspan_from_left<offsets>(ins, i)...);
        }
    }
}

#if defined(_OPENMP) && defined(REAL_GCC)

template <size_t brank, typename func_t, size_t offset, size_t... offsets,
          mdspan_c in_t, mdspan_c... ins_t>
inline void batch_impl_cpump(func_t &&func,
                             std::index_sequence<offset, offsets...>, in_t &&in,
                             ins_t &&...ins) {
    static_assert(sizeof...(offsets) == sizeof...(ins),
                  "Number of offsets must match number of inputs.");

    using index_t = typename std::remove_cvref_t<in_t>::index_type;

    if constexpr (brank == 0) {
        std::forward<func_t>(func)(in, ins...);

    } else {
#pragma omp parallel for
        for (index_t i = 0; i < in.extent(offset); i++) {
            batch_impl_none<brank - 1>(
                std::forward<func_t>(func),
                std::index_sequence<offset, offsets...>{},
                submdspan_from_left<offset>(in, i),
                submdspan_from_left<offsets>(ins, i)...);
        }
    }
}

#endif

template <MPMode mpmode, size_t brank, typename func_t, size_t... offsets,
          mdspan_c... ins_t>
inline constexpr void batch_impl(func_t &&func, std::index_sequence<offsets...>,
                                 ins_t &&...ins) {
    if constexpr (mpmode == MPMode::CPUMP) {
        batch_impl_cpump<brank>(std::forward<func_t>(func),
                                std::index_sequence<offsets...>{}, ins...);

    } else {
        batch_impl_none<brank>(std::forward<func_t>(func),
                               std::index_sequence<offsets...>{}, ins...);
    }
}

} // namespace detail

template <typename dtype, extents_c exts_t>
[[nodiscard]] inline constexpr auto create_data(exts_t &&exts) noexcept {
    using exts_base_t = std::remove_cvref_t<exts_t>;

    if constexpr (exts_base_t::rank() == 0) {
        return dtype{};

    } else {
        return mdarray<dtype, exts_base_t>{std::forward<exts_t>(exts)};
    }
}

template <typename dtype = void, size_t... offsets, size_t... uranks,
          extents_c uout_exts_t, mdspan_c... ins_t>
[[nodiscard]] inline constexpr auto
create_out(std::index_sequence<offsets...>, std::index_sequence<uranks...>,
           uout_exts_t &&uout_exts, ins_t &&...ins) noexcept {
    static_assert(sizeof...(offsets) == sizeof...(ins_t) + 1,
                  "Number of offsets must be number of inputs + 1.");
    static_assert(sizeof...(uranks) == sizeof...(ins_t),
                  "Number of uranks must match number of inputs.");

    using value_t =
        std::conditional_t<std::is_void_v<dtype>,
                           common_type_t<value_type_t<ins_t>...>, dtype>;

    constexpr auto ofst = std::array{offsets...};
    constexpr auto ur = std::array{uranks...};
    constexpr auto br = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array{
            (std::tuple_element_t<
                 Is, std::tuple<std::remove_cvref_t<ins_t>...>>::rank() -
             ur[Is])...};
    }(std::make_index_sequence<sizeof...(ins_t)>{});

    auto ins_tuple = std::forward_as_tuple(ins...);

    const auto bexts = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return broadcast(slice_with_offset<ofst[Is], br[Is]>(
            std::get<Is>(ins_tuple).extents())...);
    }(std::make_index_sequence<sizeof...(ins_t)>{});

    constexpr size_t uout_offset = ofst[sizeof...(ins_t)];

    return create_data<value_t>(
        concatenate(slice_from_left<uout_offset>(uout_exts), bexts,
                    slice_from_right<std::remove_cvref_t<uout_exts_t>::rank() -
                                     uout_offset>(uout_exts)));
}

template <typename dtype = void, size_t... uranks, extents_c uout_exts_t,
          mdspan_c... ins_t>
[[nodiscard]] inline constexpr auto create_out(std::index_sequence<uranks...>,
                                               uout_exts_t &&uout_exts,
                                               ins_t &&...ins) noexcept {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return create_out<dtype>(std::index_sequence<((void)Is, 0)...>{},
                                 std::index_sequence<uranks...>{},
                                 std::forward<uout_exts_t>(uout_exts),
                                 std::forward<ins_t>(ins)...);
    }(std::make_index_sequence<sizeof...(ins_t) + 1>{});
}

template <typename dtype = void, extents_c uout_exts_t, mdspan_c... ins_t>
[[nodiscard]] inline constexpr auto create_out(uout_exts_t &&uout_exts,
                                               ins_t &&...ins) noexcept {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return create_out<dtype>(std::index_sequence<((void)Is, 0)...>{},
                                 std::forward<uout_exts_t>(uout_exts),
                                 std::forward<ins_t>(ins)...);
    }(std::make_index_sequence<sizeof...(ins_t)>{});
}

template <typename dtype = void, size_t... offsets, size_t... uranks,
          extents_tuple_c uout_exts_tuple_t, mdspan_c... ins_t>
[[nodiscard]] inline constexpr auto
create_outs(std::index_sequence<offsets...>, std::index_sequence<uranks...>,
            uout_exts_tuple_t &&uout_exts_tuple, ins_t &&...ins) noexcept {
    constexpr size_t outs_num =
        std::tuple_size_v<std::remove_cvref_t<uout_exts_tuple_t>>;

    static_assert(
        sizeof...(offsets) == sizeof...(ins_t) + outs_num,
        "Number of offsets must be number of inputs + number of outputs.");
    static_assert(sizeof...(uranks) == sizeof...(ins_t),
                  "Number of uranks must match number of inputs.");

    using value_t =
        std::conditional_t<std::is_void_v<dtype>,
                           common_type_t<value_type_t<ins_t>...>, dtype>;

    constexpr auto ofst = std::array{offsets...};
    constexpr auto ur = std::array{uranks...};
    constexpr auto br = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array{
            (std::tuple_element_t<
                 Is, std::tuple<std::remove_cvref_t<ins_t>...>>::rank() -
             ur[Is])...};
    }(std::make_index_sequence<sizeof...(ins_t)>{});

    auto ins_tuple = std::forward_as_tuple(ins...);

    const auto bexts = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return broadcast(slice_with_offset<ofst[Is], br[Is]>(
            std::get<Is>(ins_tuple).extents())...);
    }(std::make_index_sequence<sizeof...(ins_t)>{});

    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::tuple{create_data<value_t>(concatenate(
            slice_from_left<ofst[sizeof...(ins_t) + Is]>(
                std::get<Is>(uout_exts_tuple)),
            bexts,
            slice_from_right<
                std::tuple_element_t<Is, uout_exts_tuple_t>::rank() -
                ofst[sizeof...(ins_t) + Is]>(
                std::get<Is>(uout_exts_tuple))))...};
    }(std::make_index_sequence<outs_num>{});
}

template <typename dtype = void, size_t... uranks,
          extents_tuple_c uout_exts_tuple_t, mdspan_c... ins_t>
[[nodiscard]] inline constexpr auto
create_outs(std::index_sequence<uranks...>, uout_exts_tuple_t &&uout_exts_tuple,
            ins_t &&...ins) noexcept {
    constexpr size_t outs_num =
        std::tuple_size_v<std::remove_cvref_t<uout_exts_tuple_t>>;

    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return create_data<dtype>(
            std::index_sequence<((void)Is, 0)...>{},
            std::index_sequence<uranks...>{},
            std::forward<uout_exts_tuple_t>(uout_exts_tuple),
            std::forward<ins_t>(ins)...);
    }(std::make_index_sequence<sizeof...(ins_t) + outs_num>{});
}

template <typename dtype = void, extents_tuple_c uout_exts_tuple_t,
          mdspan_c... ins_t>
[[nodiscard]] inline constexpr auto
create_outs(uout_exts_tuple_t &&uout_exts_tuple, ins_t &&...ins) noexcept {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return create_data<dtype>(
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<uout_exts_tuple_t>(uout_exts_tuple),
            std::forward<ins_t>(ins)...);
    }(std::make_index_sequence<sizeof...(ins_t)>{});
}

// TODO: remove under code
// template <typename Func, size_t... offsets, size_t... uranks, mdspan_c...
// ins_t>
//     requires(sizeof...(offsets) == sizeof...(ins_t) &&
//              sizeof...(uranks) == sizeof...(ins_t))
// inline constexpr void batch(Func &&func, std::index_sequence<offsets...>,
//                             std::index_sequence<uranks...>,
//                             const MPMode mpmode,
//                             const ins_t &...ins) noexcept {
//     constexpr auto ofst = std::array{offsets...};
//     constexpr auto ur = std::array{uranks...};
//     constexpr auto br = [&]<size_t... Is>(std::index_sequence<Is...>) {
//         return std::array{
//             (std::tuple_element_t<Is, std::tuple<ins_t...>>::rank() -
//             ofst[Is] -
//              ur[Is])...};
//     }(std::make_index_sequence<sizeof...(ins_t)>{});

//     constexpr bool no_branks = [&]<size_t... Is>(std::index_sequence<Is...>)
//     {
//         return ((br[Is] == 0) && ...);
//     }(std::make_index_sequence<sizeof...(ins_t)>{});

//     constexpr bool all_same_static_bexts =
//         [&]<size_t... Is>(std::index_sequence<Is...>) {
//             return same(slice_with_offset<ofst[Is], br[Is]>(
//                 typename std::tuple_element_t<
//                     Is, std::tuple<ins_t...>>::extents_type{})...);
//         }(std::make_index_sequence<sizeof...(ins_t)>{});

//     if constexpr (no_branks) {
//         // case 1: directly pass to unit function
//         std::forward<Func>(func)(ins...);
//         return;

//     } else if constexpr (all_same_static_bexts) {
//         // case 2: if static bexts are all same
//         constexpr bool is_flattable = [&]<size_t... Is>(
//                                           std::index_sequence<Is...>) {
//             return ((std::tuple_element_t<
//                          Is, std::tuple<ins_t...>>::rank_dynamic() == 0 &&
//                      is_always_reshapable<
//                          std::tuple_element_t<Is, std::tuple<ins_t...>>>())
//                          &&
//                     ...);
//         }(std::make_index_sequence<sizeof...(ins_t)>{});

//         if constexpr (is_flattable) {
//             // case 2-1: batch with flattening
//             constexpr size_t static_bsize =
//                 static_size(slice_with_offset<ofst[0], br[0]>(
//                     typename std::tuple_element_t<
//                         0, std::tuple<ins_t...>>::extents_type{}));

//             auto ins_tuple = std::forward_as_tuple(ins...);

//             [&]<size_t... Is>(std::index_sequence<Is...>) {
//                 detail::batch_impl<1>(
//                     std::forward<Func>(func),
//                     std::index_sequence<offsets...>{}, mpmode,
//                     static_reshape(
//                         std::get<Is>(ins_tuple),
//                         concatenate(
//                             slice_from_left<ofst[Is]>(
//                                 std::get<Is>(ins_tuple).extents()),
//                             extents<size_t, static_bsize>{static_bsize},
//                             slice_from_right<ur[Is]>(
//                                 std::get<Is>(ins_tuple).extents())))...);
//             }(std::make_index_sequence<sizeof...(ins_t)>{});
//             return;

//         } else {
//             // case 2-2: batch without flattening
//             detail::batch_impl<br[0]>(std::forward<Func>(func),
//                                       std::index_sequence<offsets...>{},
//                                       mpmode, ins...);
//             return;
//         }

//     } else {
//         // case 3: if dynamic bexts are all same
//         auto ins_tuple = std::forward_as_tuple(ins...);

//         constexpr bool possibly_same_bexts =
//             [&]<size_t... Is>(std::index_sequence<Is...>) {
//                 return ((br[Is] == br[0]) && ...);
//             }(std::make_index_sequence<sizeof...(ins_t)>{});

//         if constexpr (possibly_same_bexts) {
//             const auto same_bexts =
//                 [&]<size_t... Is>(std::index_sequence<Is...>) {
//                     return same(slice_with_offset<ofst[Is], br[Is]>(
//                         std::get<Is>(ins_tuple).extents())...);
//                 }(std::make_index_sequence<sizeof...(ins_t)>{});

//             if (same_bexts) [[likely]] {
//                 // If all bexts are same, broadcasting is not required.

//                 const bool is_flattable =
//                     [&]<size_t... Is>(std::index_sequence<Is...>) {
//                         return (is_reshapable(std::get<Is>(ins_tuple))
//                         &&
//                                 ...);
//                     }(std::make_index_sequence<sizeof...(ins_t)>{});

//                 if (is_flattable) [[likely]] {
//                     // case 3-1: batch with flattening
//                     const auto bexts = slice_with_offset<ofst[0], br[0]>(
//                         std::get<0>(ins_tuple).extents());

//                     constexpr size_t static_bsize = static_size(bexts);
//                     const size_t bsize = size(bexts);

//                     [&]<size_t... Is>(std::index_sequence<Is...>) {
//                         detail::batch_impl<1>(
//                             std::forward<Func>(func),
//                             std::index_sequence<offsets...>{}, mpmode,
//                             reshape(
//                                 std::get<Is>(ins_tuple),
//                                 concatenate(
//                                     slice_from_left<ofst[Is]>(
//                                         std::get<Is>(ins_tuple).extents()),
//                                     extents<size_t, static_bsize>{bsize},
//                                     slice_from_right<ur[Is]>(
//                                         std::get<Is>(ins_tuple)
//                                             .extents())))...);
//                     }(std::make_index_sequence<sizeof...(ins_t)>{});
//                     return;

//                 } else {
//                     // case 3-2: batch without flattening
//                     detail::batch_impl<br[0]>(std::forward<Func>(func),
//                                               std::index_sequence<offsets...>{},
//                                               mpmode, ins...);
//                     return;
//                 }
//             }
//         }

//         // case 4: batch with broadcasting
//         const auto bexts = [&]<size_t... Is>(std::index_sequence<Is...>) {
//             return broadcast(slice_with_offset<ofst[Is], br[Is]>(
//                 std::get<Is>(ins_tuple).extents())...);
//         }(std::make_index_sequence<sizeof...(ins_t)>{});

//         [&]<size_t... Is>(std::index_sequence<Is...>) {
//             detail::batch_impl<bexts.rank()>(
//                 std::forward<Func>(func), std::index_sequence<offsets...>{},
//                 mpmode,
//                 broadcast_to<ofst[Is], br[Is]>(std::get<Is>(ins_tuple),
//                                                bexts)...);
//         }(std::make_index_sequence<sizeof...(ins_t)>{});
//     }
// }

template <MPMode mpmode, typename func_t, size_t... offsets, size_t... uranks,
          mdspan_c... ins_t>
inline constexpr void batch(func_t &&func, std::index_sequence<offsets...>,
                            std::index_sequence<uranks...>, ins_t &&...ins) {
    static_assert(sizeof...(offsets) == sizeof...(ins_t));
    static_assert(sizeof...(uranks) == sizeof...(ins_t));

    constexpr auto fr = std::array{std::remove_cvref_t<ins_t>::rank()...};
    constexpr auto ofst = std::array{offsets...};
    constexpr auto ur = std::array{uranks...};
    constexpr auto br = [&] {
        std::array<std::size_t, sizeof...(ins_t)> br{};
        for (size_t i = 0; i < br.size(); i++) {
            br[i] = fr[i] - ofst[i] - ur[i];
        }
        return br;
    }();

    auto ins_tuple = std::forward_as_tuple(ins...);

    constexpr bool no_branks = [&] {
        for (auto v : br) {
            if (v != 0) {
                return false;
            }
        }
        return true;
    }();

    constexpr bool all_same_static_bexts = [&]<size_t... Is>(
                                               std::index_sequence<Is...>) {
        return same(slice_with_offset<ofst[Is], br[Is]>(
            typename std::tuple_element_t<
                Is,
                std::tuple<std::remove_cvref_t<ins_t>...>>::extents_type{})...);
    }(std::make_index_sequence<sizeof...(ins_t)>{});

    if constexpr (no_branks) {
        std::forward<func_t>(func)(ins...);
        return;

    } else if constexpr (all_same_static_bexts) {
#ifdef MDTENSOR_USE_AUTOFLAT
        constexpr bool is_flattable = [&]<size_t... Is>(
                                          std::index_sequence<Is...>) {
            return ((std::tuple_element_t<Is, std::tuple<std::remove_cvref_t<
                                                  ins_t>...>>::rank_dynamic() ==
                         0 &&
                     is_always_reshapable<std::tuple_element_t<
                         Is, std::tuple<std::remove_cvref_t<ins_t>...>>>()) &&
                    ...);
        }(std::make_index_sequence<sizeof...(ins_t)>{});

        if constexpr (is_flattable) {
            constexpr size_t static_bsize =
                static_size(slice_with_offset<ofst[0], br[0]>(
                    typename std::tuple_element_t<
                        0, std::tuple<std::remove_cvref_t<ins_t>...>>::
                        extents_type{}));

            [&]<size_t... Is>(std::index_sequence<Is...>) {
                detail::batch_impl<mpmode, 1>(
                    std::forward<func_t>(func),
                    std::index_sequence<offsets...>{},
                    static_reshape(
                        std::get<Is>(ins_tuple),
                        concatenate(
                            slice_from_left<ofst[Is]>(
                                std::get<Is>(ins_tuple).extents()),
                            extents<size_t, static_bsize>{static_bsize},
                            slice_from_right<ur[Is]>(
                                std::get<Is>(ins_tuple).extents())))...);
            }(std::make_index_sequence<sizeof...(ins_t)>{});
            return;
        }
#endif

        detail::batch_impl<mpmode, br[0]>(std::forward<func_t>(func),
                                          std::index_sequence<offsets...>{},
                                          ins...);

    } else {
        const auto bexts = [&]<size_t... Is>(std::index_sequence<Is...>) {
            return broadcast(slice_with_offset<ofst[Is], br[Is]>(
                std::get<Is>(ins_tuple).extents())...);
        }(std::make_index_sequence<sizeof...(ins_t)>{});

        [&]<size_t... Is>(std::index_sequence<Is...>) {
            detail::batch_impl<mpmode, decltype(bexts)::rank()>(
                std::forward<func_t>(func), std::index_sequence<offsets...>{},
                broadcast_to<ofst[Is], br[Is]>(std::get<Is>(ins_tuple),
                                               bexts)...);
        }(std::make_index_sequence<sizeof...(ins_t)>{});
    }
}

template <MPMode mpmode, typename func_t, size_t... uranks, mdspan_c... ins_t>
inline constexpr void batch(func_t &&func, std::index_sequence<uranks...>,
                            ins_t &&...ins) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        batch<mpmode>(std::forward<func_t>(func),
                      std::index_sequence<((void)Is, 0)...>{},
                      std::index_sequence<uranks...>{}, ins...);
    }(std::make_index_sequence<sizeof...(ins_t)>{});
}

template <MPMode mpmode, typename func_t, mdspan_c... ins_t>
inline constexpr void batch(func_t &&func, ins_t &&...ins) {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        batch<mpmode>(std::forward<func_t>(func),
                      std::index_sequence<((void)Is, 0)...>{}, ins...);
    }(std::make_index_sequence<sizeof...(ins_t)>{});
}

template <MPMode mpmode, typename dtype, typename func_t, size_t... offsets,
          size_t... uranks, extents_c uout_exts_t, mdspan_c... ins_t>
[[nodiscard]] inline constexpr auto
batch_out(func_t &&func, std::index_sequence<offsets...>,
          std::index_sequence<uranks...>, uout_exts_t &&uout_exts,
          ins_t &&...ins) {
    static_assert(sizeof...(uranks) == sizeof...(ins_t));
    static_assert(sizeof...(offsets) == sizeof...(ins_t) + 1);

    auto out = create_out<dtype>(
        std::index_sequence<offsets...>{}, std::index_sequence<uranks...>{},
        std::forward<uout_exts_t>(uout_exts), std::forward<ins_t>(ins)...);

    batch<mpmode>(std::forward<func_t>(func), std::index_sequence<offsets...>{},
                  std::index_sequence<uranks..., uout_exts.rank()>{},
                  std::forward<ins_t>(ins)..., to_mdspan(out));

    return out;
}

template <MPMode mpmode, typename dtype, typename func_t, size_t... offsets,
          size_t... uranks, extents_tuple_c uout_exts_tuple_t,
          mdspan_c... ins_t>
[[nodiscard]] inline constexpr auto
batch_out(func_t &&func, std::index_sequence<offsets...>,
          std::index_sequence<uranks...>, uout_exts_tuple_t &&uout_exts_tuple,
          ins_t &&...ins) {
    constexpr size_t outs_num =
        std::tuple_size_v<std::remove_cvref_t<uout_exts_tuple_t>>;

    static_assert(sizeof...(uranks) == sizeof...(ins_t));
    static_assert(sizeof...(offsets) == sizeof...(ins_t) + outs_num);

    auto outs = create_outs<dtype>(
        std::index_sequence<offsets...>{}, std::index_sequence<uranks...>{},
        std::forward<uout_exts_tuple_t>(uout_exts_tuple),
        std::forward<ins_t>(ins)...);

    [&]<size_t... Is>(std::index_sequence<Is...>) {
        batch<mpmode>(
            std::forward<func_t>(func), std::index_sequence<offsets...>{},
            std::index_sequence<
                uranks...,
                std::tuple_element_t<
                    Is, std::remove_cvref_t<uout_exts_tuple_t>>::rank()...>{},
            std::forward<ins_t>(ins)..., to_mdspan(std::get<Is>(outs))...);
    }(std::make_index_sequence<outs_num>{});

    return outs;
}

template <MPMode mpmode, typename dtype, typename func_t, size_t... uranks,
          extents_info_c uout_info_t, mdspan_c... ins_t>
[[nodiscard]] inline constexpr auto
batch_out(func_t &&func, std::index_sequence<uranks...>,
          uout_info_t &&uout_info, ins_t &&...ins) {
    constexpr size_t uout_len = [] {
        using uout_exts_base_t = std::remove_cvref_t<uout_info_t>;

        if constexpr (is_extents_v<uout_exts_base_t>) {
            return 1;

        } else {
            return std::tuple_size_v<uout_exts_base_t>;
        }
    }();

    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return batch_out<mpmode, dtype>(
            std::forward<func_t>(func), std::index_sequence<((void)Is, 0)...>{},
            std::index_sequence<uranks...>{},
            std::forward<uout_info_t>(uout_info), std::forward<ins_t>(ins)...);
    }(std::make_index_sequence<sizeof...(ins_t) + uout_len>{});
}

template <MPMode mpmode, typename dtype, typename func_t,
          extents_info_c uout_info_t, mdspan_c... ins_t>
[[nodiscard]] inline constexpr auto
batch_out(func_t &&func, uout_info_t &&uout_info, ins_t &&...ins) {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return batch_out<mpmode, dtype>(
            std::forward<func_t>(func), std::index_sequence<((void)Is, 0)...>{},
            std::forward<uout_info_t>(uout_info), std::forward<ins_t>(ins)...);
    }(std::make_index_sequence<sizeof...(ins_t)>{});
}

} // namespace core
} // namespace mdtensor
