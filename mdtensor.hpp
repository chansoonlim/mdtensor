namespace std::experimental {
    using namespace std; // NOTE: fix submdspan including error
}
#ifndef MDTENSOR_SINGLE_HEADER_INCLUDE_GUARD_
#define MDTENSOR_SINGLE_HEADER_INCLUDE_GUARD_

//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/mdtensor.hpp
/**
 * @file
 * @brief Top-level public API header for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/creation.hpp
/**
 * @file
 * @brief Creation module header aggregation for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/arange.hpp
/**
 * @file
 * @brief Range generation utilities for mdtensor (arange).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/core.hpp
/**
 * @file
 * @brief Core utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


#include <cmath>

//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/broadcast.hpp
/**
 * @file
 * @brief Broadcasting utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


#include <optional>
#include <tuple>

//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/submdspan.hpp
/**
 * @file
 * @brief Submdspan utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/convert.hpp
/**
 * @file
 * @brief Conversion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/type.hpp
/**
 * @file
 * @brief Type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


#include <concepts>

#ifndef MDSPAN_SINGLE_HEADER_INCLUDE_GUARD_ // NOTE: for godbolt test
#include <experimental/mdarray>
#include <experimental/mdspan> // TODO: Remove when C++23 std::mdspan supports
#endif

namespace std::experimental {

// dims: will be included in C++23
// (https://en.cppreference.com/w/cpp/container/mdspan/extents)
template <size_t Rank, class IndexType = size_t>
using dims = dextents<IndexType, Rank>;

static_assert(dynamic_extent == std::dynamic_extent);

constexpr auto dyn = dynamic_extent;

} // namespace std::experimental

namespace mdtensor {

using namespace std::experimental;

#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
#define REAL_GCC __GNUC__ // probably
#endif

// ----------------------------------------------------------------------

template <typename T>
concept arithmetic_c = std::is_arithmetic_v<std::remove_cvref_t<T>>;

// ----------------------------------------------------------------------

namespace detail {

template <typename T> struct is_extents_impl : std::false_type {};

template <typename IndexType, size_t... Extents>
struct is_extents_impl<std::experimental::extents<IndexType, Extents...>>
    : std::true_type {};

} // namespace detail

template <typename T>
struct is_extents : detail::is_extents_impl<std::remove_cvref_t<T>> {};

template <typename T> constexpr bool is_extents_v = is_extents<T>::value;

template <typename T>
concept extents_c = is_extents_v<T>;

// ----------------------------------------------------------------------

namespace detail {

template <typename T> struct is_extents_tuple_impl : std::false_type {};

template <typename... ExtentsTypes>
struct is_extents_tuple_impl<std::tuple<ExtentsTypes...>>
    : std::conjunction<is_extents<ExtentsTypes>...> {};

} // namespace detail

template <typename T>
struct is_extents_tuple
    : detail::is_extents_tuple_impl<std::remove_cvref_t<T>> {};

template <typename T>
constexpr bool is_extents_tuple_v = is_extents_tuple<T>::value;

template <typename T>
concept extents_tuple_c = is_extents_tuple_v<T>;

// ----------------------------------------------------------------------

template <typename T>
concept extents_info_c = extents_c<T> || extents_tuple_c<T>;

// ----------------------------------------------------------------------

namespace detail {

template <typename T> struct is_mdspan_impl : std::false_type {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename AccessorType>
struct is_mdspan_impl<std::experimental::mdspan<ElementType, ExtentsType,
                                                LayoutType, AccessorType>>
    : std::true_type {};

} // namespace detail

template <typename T>
struct is_mdspan : detail::is_mdspan_impl<std::remove_cvref_t<T>> {};

template <typename T> constexpr bool is_mdspan_v = is_mdspan<T>::value;

template <typename T>
concept mdspan_c = requires {
    typename std::remove_cvref_t<T>::value_type;
    typename std::remove_cvref_t<T>::element_type;
    typename std::remove_cvref_t<T>::extents_type;
    typename std::remove_cvref_t<T>::layout_type;
    typename std::remove_cvref_t<T>::accessor_type;
} && is_mdspan_v<T>;

// ----------------------------------------------------------------------

namespace detail {

template <typename T> struct is_mdarray_impl : std::false_type {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename ContainerType>
struct is_mdarray_impl<std::experimental::mdarray<ElementType, ExtentsType,
                                                  LayoutType, ContainerType>>
    : std::true_type {};

} // namespace detail

template <typename T>
struct is_mdarray : detail::is_mdarray_impl<std::remove_cvref_t<T>> {};

template <typename T> constexpr bool is_mdarray_v = is_mdarray<T>::value;

template <typename T>
concept mdarray_c = is_mdarray_v<T>;

// ----------------------------------------------------------------------

template <typename T> constexpr bool is_md_v = mdspan_c<T> || mdarray_c<T>;

template <typename T>
concept md_c = is_md_v<T>;

// ----------------------------------------------------------------------

namespace core {

template <extents_c exts_t>
[[nodiscard]] inline constexpr size_t
static_size(const exts_t &exts = exts_t{}) noexcept {
    if constexpr (exts_t::rank() == 0) {
        return 0;

    } else if constexpr (exts_t::rank_dynamic() == 0) {
        return []<size_t... Is>(std::index_sequence<Is...>) {
            return (exts_t::static_extent(Is) * ...);
        }(std::make_index_sequence<exts_t::rank()>{});

    } else {
        return dyn;
    }
}

template <extents_c exts_t>
[[nodiscard]] inline constexpr size_t
size(const exts_t &exts = exts_t{}) noexcept {
    if constexpr (exts_t::rank() == 0) {
        return 0;

    } else if constexpr (exts_t::rank_dynamic() == 0) {
        return []<size_t... Is>(std::index_sequence<Is...>) {
            return (exts_t::static_extent(Is) * ...);
        }(std::make_index_sequence<exts_t::rank()>{});

    } else {
        return [&exts]<size_t... Is>(std::index_sequence<Is...>) {
            return (exts.extent(Is) * ...);
        }(std::make_index_sequence<exts_t::rank()>{});
    }
}

template <extents_c in_t>
[[nodiscard]] inline constexpr bool same(const in_t &in = in_t{}) noexcept {
    return true;
}

template <extents_c in1_t, extents_c in2_t, extents_c... ins_t>
[[nodiscard]] inline constexpr bool same(const in1_t &in1 = in1_t{},
                                         const in2_t &in2 = in2_t{},
                                         const ins_t &...ins) noexcept {
    if constexpr (in1_t::rank() != in2_t::rank()) {
        return false;

    } else if constexpr (in1_t::rank_dynamic() == 0 &&
                         in2_t::rank_dynamic() == 0 &&
                         []<size_t... Is>(std::index_sequence<Is...>) {
                             return ((in1_t::static_extent(Is) !=
                                      in2_t::static_extent(Is)) ||
                                     ...);
                         }(std::make_index_sequence<in1_t::rank()>{})) {
        return false;

    } else {
        for (size_t i = 0; i < in1_t::rank(); i++) {
            if (static_cast<size_t>(in1.extent(i)) !=
                static_cast<size_t>(in2.extent(i))) {
                return false;
            }
        }
    }

    if constexpr (sizeof...(ins_t) != 0) {
        return same(in2, ins...);

    } else {
        return true;
    }
}

// ----------------------------------------------------------------------

namespace detail {

template <typename T, typename = void> struct value_type_t_impl {
    using type = std::remove_cvref_t<T>;
};

template <typename T>
struct value_type_t_impl<
    T, std::void_t<typename std::remove_cvref_t<T>::value_type>> {
    using type = typename std::remove_cvref_t<T>::value_type;
};

} // namespace detail

template <typename T>
using value_type_t = typename detail::value_type_t_impl<T>::type;

// ----------------------------------------------------------------------

namespace detail {

template <std::size_t Size> struct signed_by_size;

template <> struct signed_by_size<1> {
    using type = int8_t;
};

template <> struct signed_by_size<2> {
    using type = int16_t;
};

template <> struct signed_by_size<4> {
    using type = int32_t;
};

template <> struct signed_by_size<8> {
    using type = int64_t;
};

template <std::size_t Size>
using signed_by_size_t = typename signed_by_size<Size>::type;

template <typename T>
inline constexpr bool valid_extent_index_v =
    std::integral<std::remove_cvref_t<T>> &&
    !std::same_as<std::remove_cvref_t<T>, bool> &&
    (sizeof(std::remove_cvref_t<T>) == 1 ||
     sizeof(std::remove_cvref_t<T>) == 2 ||
     sizeof(std::remove_cvref_t<T>) == 4 ||
     sizeof(std::remove_cvref_t<T>) == 8);

template <typename... Ts> struct common_index_type_impl {
    // no type
};

template <typename T>
    requires valid_extent_index_v<T>
struct common_index_type_impl<T> {
    using type = std::remove_cvref_t<T>;
};

template <typename T1, typename T2>
    requires(valid_extent_index_v<T1> && valid_extent_index_v<T2> &&
             (std::is_signed_v<std::remove_cvref_t<T1>> ==
              std::is_signed_v<std::remove_cvref_t<T2>>))
struct common_index_type_impl<T1, T2> {
    using type =
        std::conditional_t<(sizeof(std::remove_cvref_t<T1>) >=
                            sizeof(std::remove_cvref_t<T2>)),
                           std::remove_cvref_t<T1>, std::remove_cvref_t<T2>>;
};

template <typename S, typename U>
    requires(valid_extent_index_v<S> && valid_extent_index_v<U> &&
             std::is_signed_v<std::remove_cvref_t<S>> &&
             std::is_unsigned_v<std::remove_cvref_t<U>> &&
             (sizeof(std::remove_cvref_t<U>) < 8))
struct common_index_type_impl<S, U> {
  private:
    static constexpr std::size_t size =
        (sizeof(std::remove_cvref_t<S>) > sizeof(std::remove_cvref_t<U>))
            ? sizeof(std::remove_cvref_t<S>)
            : sizeof(std::remove_cvref_t<U>) * 2;

  public:
    using type = signed_by_size_t<size>;
};

template <typename U, typename S>
    requires(valid_extent_index_v<U> && valid_extent_index_v<S> &&
             std::is_unsigned_v<std::remove_cvref_t<U>> &&
             std::is_signed_v<std::remove_cvref_t<S>> &&
             (sizeof(std::remove_cvref_t<U>) < 8))
struct common_index_type_impl<U, S>
    : common_index_type_impl<std::remove_cvref_t<S>, std::remove_cvref_t<U>> {};

template <typename T1, typename T2, typename... Ts>
    requires requires { typename common_index_type_impl<T1, T2>::type; }
struct common_index_type_impl<T1, T2, Ts...> {
  public:
    using type = typename common_index_type_impl<
        typename common_index_type_impl<T1, T2>::type, Ts...>::type;
};

} // namespace detail

template <typename... Ts>
using common_index_type_t =
    typename detail::common_index_type_impl<Ts...>::type;

// ----------------------------------------------------------------------

namespace detail {

template <typename T> struct unwrap_optional {
    using type = std::remove_cvref_t<T>;
};

template <typename T> struct unwrap_optional<std::optional<T>> {
    using type = std::remove_cvref_t<T>;
};

template <typename T>
using data_arg_t = typename unwrap_optional<std::remove_cvref_t<T>>::type;

template <typename... Ts> struct filter_nullopt;

template <> struct filter_nullopt<> {
    using type = std::tuple<>;
};

template <typename T, typename... Ts> struct filter_nullopt<T, Ts...> {
  private:
    using arg_t = data_arg_t<T>;
    using tail_t = typename filter_nullopt<Ts...>::type;

  public:
    using type = std::conditional_t<
        std::same_as<std::remove_cvref_t<T>, std::nullopt_t>, tail_t,
        decltype(std::tuple_cat(std::declval<std::tuple<arg_t>>(),
                                std::declval<tail_t>()))>;
};

template <typename T>
inline constexpr bool data_bool_v = std::same_as<std::remove_cvref_t<T>, bool>;

template <typename T>
inline constexpr bool data_fpoint_v =
    std::floating_point<std::remove_cvref_t<T>>;

template <typename T>
inline constexpr bool data_integer_v =
    std::integral<std::remove_cvref_t<T>> && !data_bool_v<T>;

template <typename T>
inline constexpr bool data_scalar_v =
    data_bool_v<T> || data_integer_v<T> || data_fpoint_v<T>;

template <typename T1, typename T2> struct common_data_pair_impl {
    // no type
};

// bool + bool -> bool
template <typename T1, typename T2>
    requires(data_bool_v<T1> && data_bool_v<T2>)
struct common_data_pair_impl<T1, T2> {
    using type = bool;
};

// bool + T -> T
template <typename B, typename T>
    requires(data_bool_v<B> && !data_bool_v<T> && data_scalar_v<T>)
struct common_data_pair_impl<B, T> {
    using type = std::remove_cvref_t<T>;
};

// T + bool -> T
template <typename T, typename B>
    requires(!data_bool_v<T> && data_scalar_v<T> && data_bool_v<B>)
struct common_data_pair_impl<T, B> {
    using type = std::remove_cvref_t<T>;
};

// floating + something -> floating
template <typename T1, typename T2>
    requires(!data_bool_v<T1> && !data_bool_v<T2> && data_scalar_v<T1> &&
             data_scalar_v<T2> && (data_fpoint_v<T1> || data_fpoint_v<T2>))
struct common_data_pair_impl<T1, T2> {
  private:
    using lhs_t = std::remove_cvref_t<T1>;
    using rhs_t = std::remove_cvref_t<T2>;

  public:
    using type = std::conditional_t<
        data_fpoint_v<T1> && data_fpoint_v<T2>,
        std::conditional_t<(sizeof(lhs_t) >= sizeof(rhs_t)), lhs_t, rhs_t>,
        std::conditional_t<data_fpoint_v<T1>, lhs_t, rhs_t>>;
};

// integer + integer -> common_index_type_t
template <typename T1, typename T2>
    requires(data_integer_v<T1> && data_integer_v<T2> &&
             requires {
                 typename common_index_type_t<std::remove_cvref_t<T1>,
                                              std::remove_cvref_t<T2>>;
             })
struct common_data_pair_impl<T1, T2> {
    using type =
        common_index_type_t<std::remove_cvref_t<T1>, std::remove_cvref_t<T2>>;
};

template <typename... Ts> struct data_promote_impl {
    // no type
};

template <typename T>
    requires data_scalar_v<T>
struct data_promote_impl<T> {
    using type = std::remove_cvref_t<T>;
};

template <typename T1, typename T2>
struct data_promote_impl<T1, T2>
    : common_data_pair_impl<std::remove_cvref_t<T1>, std::remove_cvref_t<T2>> {
};

template <typename T1, typename T2, typename... Ts>
    requires requires {
        typename common_data_pair_impl<std::remove_cvref_t<T1>,
                                       std::remove_cvref_t<T2>>::type;
    }
struct data_promote_impl<T1, T2, Ts...> {
  public:
    using type = typename data_promote_impl<
        typename common_data_pair_impl<std::remove_cvref_t<T1>,
                                       std::remove_cvref_t<T2>>::type,
        Ts...>::type;
};

template <typename Tuple> struct common_data_type_impl {
    // no type
};

template <typename... Ts>
struct common_data_type_impl<std::tuple<Ts...>> : data_promote_impl<Ts...> {};

} // namespace detail

template <typename... Ts>
using common_data_type_t = typename detail::common_data_type_impl<
    typename detail::filter_nullopt<Ts...>::type>::type;

// ----------------------------------------------------------------------

} // namespace core

template <typename T, extents_c extent_t>
using mdarray = std::conditional_t<
    extent_t::rank_dynamic() == 0,
    std::experimental::mdarray<T, extent_t, layout_right,
                               std::array<T, core::static_size<extent_t>()>>,
    std::experimental::mdarray<T, extent_t, layout_right, std::vector<T>>>;

template <size_t start, size_t end>
using slice = std::experimental::strided_slice<
    std::integral_constant<size_t, start>,
    std::integral_constant<size_t, end - start>,
    std::integral_constant<size_t, 1>>;

enum class MPMode : uint8_t {
    NONE,  // No parallelization
    SIMD,  // SIMD parallelization
    CPUMP, // CPU multi-processing with OpenMP
};

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/type.hpp

namespace mdtensor {
namespace core {

template <typename in_t>
[[nodiscard]] inline constexpr auto to_mdspan(in_t &&in) noexcept {
    using in_base_t = std::remove_reference_t<in_t>;

    if constexpr (mdspan_c<in_base_t>) {
        return std::forward<in_t>(in);

    } else if constexpr (mdarray_c<in_base_t>) {
        return in.to_mdspan();

    } else if constexpr (requires { in.to_mdspan(); }) {
        return in.to_mdspan();

    } else {
        auto exts = extents<size_t>{};
        return mdspan<in_base_t, decltype(exts)>{&in, exts};
    }
}

template <md_c in_t>
[[nodiscard]] inline constexpr auto to_const(in_t &&in) noexcept {
    using in_base_t = std::remove_reference_t<in_t>;

    if constexpr (std::is_const_v<typename in_base_t::element_type>) {
        return std::forward<in_t>(in);

    } else {
        using element_t = const typename in_base_t::value_type;

        return mdspan<element_t, typename in_base_t::extents_type,
                      typename in_base_t::layout_type,
                      default_accessor<element_t>>(in.data_handle(),
                                                   in.mapping(), {});
    }
}

template <typename in_t>
[[nodiscard]] inline constexpr auto to_const_mdspan(in_t &&in) noexcept {
    return to_const(to_mdspan(std::forward<in_t>(in)));
}

template <md_c in_t>
[[nodiscard]] inline constexpr bool is_always_reshapable() noexcept {
    return in_t::is_always_unique() && in_t::is_always_exhaustive() &&
           in_t::is_always_strided();
}

template <md_c in_t>
[[nodiscard]] inline constexpr bool is_reshapable(in_t &&in) noexcept {
    return in.is_unique() && in.is_exhaustive() && in.is_strided();
}

template <extents_c exts_t, typename in_t>
[[nodiscard]] inline constexpr auto
static_reshape(in_t &&in, exts_t &&new_exts = exts_t{}) noexcept {
    const auto in_mds = to_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    static_assert(exts_t::rank_dynamic() == 0,
                  "static_reshape requires static extents.");

    static_assert(in_mds_t::rank_dynamic() == 0,
                  "static_reshape requires static mdspan.");

    static_assert(is_always_reshapable<in_mds_t>(),
                  "static_reshape requires always reshapable mdspan.");

    static_assert(core::static_size<typename in_mds_t::extents_type>() ==
                      core::static_size<exts_t>(),
                  "static_reshape requires same number of elements.");

    return mdspan<typename in_mds_t::element_type, exts_t>{
        in_mds.data_handle()};
}

template <extents_c exts_t, typename in_t>
[[nodiscard]] inline constexpr auto
reshape(in_t &&in, exts_t &&new_exts = exts_t{}) noexcept {
    const auto in_mds = to_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    assert(is_reshapable(in_mds));
    assert(core::size(in_mds.extents()) ==
           core::size(std::forward<exts_t>(new_exts)));

    return mdspan<typename in_mds_t::element_type, std::remove_cvref_t<exts_t>>{
        in_mds.data_handle(), std::forward<exts_t>(new_exts)};
}

template <typename T>
using to_mdspan_t = decltype(to_mdspan(std::declval<T>()));

template <typename T>
using to_mdcontainer_t =
    std::conditional_t<to_mdspan_t<T>::rank() == 0,
                       typename to_mdspan_t<T>::value_type,
                       mdarray<typename to_mdspan_t<T>::value_type,
                               typename to_mdspan_t<T>::extents_type>>;

} // namespace core
} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/convert.hpp

namespace mdtensor {

template <typename in_t, typename... slices_t>
[[nodiscard]] inline constexpr auto submdspan(in_t &&in,
                                              slices_t &&...slices) noexcept {
    return std::experimental::submdspan(core::to_mdspan(std::forward<in_t>(in)),
                                        std::forward<slices_t>(slices)...);
}

namespace core {

template <size_t lspace, size_t rspace, typename in_t, typename... slices_t>
[[nodiscard]] inline constexpr auto
submdspan_with_space(in_t &&in, slices_t &&...slices) noexcept {
    return [&]<size_t... Is, size_t... Js>(std::index_sequence<Is...>,
                                           std::index_sequence<Js...>) {
        return submdspan(std::forward<in_t>(in), ((void)Is, full_extent)...,
                         std::forward<slices_t>(slices)...,
                         ((void)Js, full_extent)...);
    }(std::make_index_sequence<lspace>{}, std::make_index_sequence<rspace>{});
}

template <size_t lspace = 0, typename in_t, typename... slices_t>
[[nodiscard]] inline constexpr auto
submdspan_from_left(in_t &&in, slices_t &&...slices) noexcept {
    using in_base_t = std::remove_reference_t<in_t>;

    constexpr size_t rspace =
        to_mdspan_t<in_base_t>::rank() - (lspace + sizeof...(slices_t));

    return submdspan_with_space<lspace, rspace>(
        std::forward<in_t>(in), std::forward<slices_t>(slices)...);
}

template <size_t rspace = 0, typename in_t, typename... slices_t>
[[nodiscard]] inline constexpr auto
submdspan_from_right(in_t &&in, slices_t &&...slices) noexcept {
    using in_base_t = std::remove_reference_t<in_t>;

    constexpr size_t lspace =
        to_mdspan_t<in_base_t>::rank() - (rspace + sizeof...(slices_t));

    return submdspan_with_space<lspace, rspace>(
        std::forward<in_t>(in), std::forward<slices_t>(slices)...);
}

} // namespace core
} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/submdspan.hpp

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
    using index_t = common_index_type_t<typename in1_base_t::index_type,
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
    using index_t = common_index_type_t<typename in1_base_t::index_type,
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
            common_index_type_t<typename in_base_t::index_type,
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
                           common_data_type_t<value_type_t<ins_t>...>, dtype>;

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
                           common_data_type_t<value_type_t<ins_t>...>, dtype>;

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

template <typename dtype, MPMode mpmode, typename func_t, size_t... offsets,
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

template <typename dtype, MPMode mpmode, typename func_t, size_t... offsets,
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

template <typename dtype, MPMode mpmode, typename func_t, size_t... uranks,
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
        return batch_out<dtype, mpmode>(
            std::forward<func_t>(func), std::index_sequence<((void)Is, 0)...>{},
            std::index_sequence<uranks...>{},
            std::forward<uout_info_t>(uout_info), std::forward<ins_t>(ins)...);
    }(std::make_index_sequence<sizeof...(ins_t) + uout_len>{});
}

template <typename dtype, MPMode mpmode, typename func_t,
          extents_info_c uout_info_t, mdspan_c... ins_t>
[[nodiscard]] inline constexpr auto
batch_out(func_t &&func, uout_info_t &&uout_info, ins_t &&...ins) {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return batch_out<dtype, mpmode>(
            std::forward<func_t>(func), std::index_sequence<((void)Is, 0)...>{},
            std::forward<uout_info_t>(uout_info), std::forward<ins_t>(ins)...);
    }(std::make_index_sequence<sizeof...(ins_t)>{});
}

} // namespace core
} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/broadcast.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/eigen/eigen.hpp
/**
 * @file
 * @brief Eigen interop utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


#ifdef MDTENSOR_USE_EIGEN
#include <Eigen/Dense>
#endif


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
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/eigen/eigen.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/core.hpp

namespace mdtensor {

/**
 * @brief Generate a 1-D range of values with a given step (out-of-place).
 *
 * @tparam data_t (optional) Output value type. If void, deduced from inputs.
 * @tparam start_t Arithmetic type of start.
 * @tparam stop_t Arithmetic type of stop.
 * @tparam step_t (optional) Arithmetic type of step. Default is double.
 *
 * @param start Start value (inclusive).
 * @param stop Stop value (exclusive in intent; number of elements is computed
 *             from (stop - start) / step).
 * @param step Step size. Default is 1.
 *
 * @return 1-D mdarray containing the generated range.
 *
 * @note The output length is computed as ceil((stop - start) / step).
 * @note The effective step is computed in the output value type to reduce
 *       accumulation drift when `start_t`/`step_t` differ from `data_t`.
 *
 * @warning This implementation does not currently guard against invalid input
 *          such as step == 0 or mismatched sign between (stop - start) and
 *          step. Callers should ensure the range definition is valid.
 */
template <typename data_t = void, arithmetic_c start_t, arithmetic_c stop_t,
          arithmetic_c step_t = double>
[[nodiscard]] inline constexpr auto arange(start_t &&start, stop_t &&stop,
                                           step_t &&step = (step_t)1) noexcept {
    using value_t =
        std::conditional_t<!std::is_void_v<data_t>, data_t,
                           core::common_data_type_t<start_t, stop_t>>;

    const size_t num = std::ceil((stop - start) / step);
    const value_t step_actual =
        static_cast<value_t>(start + step) - static_cast<value_t>(start);

    auto out = mdarray<value_t, mdtensor::dims<1>>{mdtensor::dims<1>{num}};

    out(0) = start;
    for (size_t i = 1; i < num; i++) {
        out(i) = out(i - 1) + step_actual;
    }

    return out;
}

/**
 * @brief Generate a 1-D range [0, stop) with step 1 (out-of-place).
 *
 * @tparam data_t (optional) Output value type. If void, deduced from input.
 * @tparam stop_t Arithmetic type of stop.
 *
 * @param stop Stop value (exclusive in intent).
 *
 * @return 1-D mdarray containing the generated range.
 *
 * @see mdtensor::arange(start, stop, step) for the general form.
 */
template <typename data_t = void, arithmetic_c stop_t>
[[nodiscard]] inline constexpr auto arange(stop_t &&stop) noexcept {
    return arange<data_t>(0, std::forward<stop_t>(stop));
}

// TODO: Develop arange that works in compile-time contexts (e.g., constexpr
// mdarray) and/or with static extents.

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/arange.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/copy.hpp
/**
 * @file
 * @brief Copy utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in_t, typename out_t>
inline constexpr void copy_impl(in_t &&in, out_t &&out) {
    out() = in();
}

} // namespace detail

/**
 * @brief Copy input to output element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input (mdspan, mdarray, scalar, etc.).
 * @param out Output (mdspan, mdarray, scalar, etc.).
 *
 * @note Equivalent to out = in in terms of array broadcasting.
 *
 * @see mdtensor::copy for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void copy_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::copy_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Copy input element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 * input.
 *
 * @param in Input (mdspan, mdarray, scalar, etc.).
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = in in terms of array broadcasting.
 *
 * @see mdtensor::copy_to for the in-place version that writes into an output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto copy(in_t &&in) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::copy_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/copy.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/copy_like.hpp
/**
 * @file
 * @brief Copy-into-like utilities for mdtensor (copy_like).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/empty_like.hpp
/**
 * @file
 * @brief Empty-like tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/empty.hpp
/**
 * @file
 * @brief Empty tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

/**
 * @brief Create an uninitialized tensor with the given extents.
 *
 * @tparam dtype Element type of the created tensor.
 * @tparam exts_t (optional) Extents type describing the tensor shape.
 *         Default is extents<uint8_t>.
 *
 * @param exts Tensor extents (shape). If omitted, creates a scalar-like tensor.
 *
 * @return A newly allocated mdarray-like tensor of type `dtype` with the given
 * extents.
 *
 * @note This function does not initialize memory. Use `zeros`, `ones`, or
 *       `full` if initialization is required.
 *
 * @see mdtensor::empty_like for creating an uninitialized tensor matching the
 * shape of an existing tensor.
 */
template <typename dtype, extents_c exts_t = extents<uint8_t>>
[[nodiscard]] inline constexpr auto empty(exts_t &&exts = exts_t{}) noexcept {
    return core::create_data<dtype>(std::forward<exts_t>(exts));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/empty.hpp

namespace mdtensor {

/**
 * @brief Create an uninitialized tensor with the same shape as the input.
 *
 * @tparam dtype (optional) Element type of the result. If void, the value type
 *         is deduced from the input.
 *
 * @param in Input tensor-like object (mdspan, mdarray, etc.).
 *
 * @return A new tensor allocated with the same extents as `in`.
 *
 * @note This function does not initialize memory. Use `zeros_like` or
 *       `full_like` if initialization is required.
 *
 * @see mdtensor::empty for creating an uninitialized tensor from explicit
 * extents.
 */
template <typename dtype = void, typename in_t>
[[nodiscard]] inline constexpr auto empty_like(in_t &&in) noexcept {
    using value_t = std::conditional_t<std::is_void_v<dtype>,
                                       core::value_type_t<in_t>, dtype>;

    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    return empty<value_t>(in_mds.extents());
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/empty_like.hpp

namespace mdtensor {

/**
 * @brief Allocate an output like `in1` and copy `in2` into it (out-of-place).
 *
 * @tparam dtype (optional) Output value type. If void, deduced by
 *         empty_like.
 * @tparam mpmode (optional) Parallel execution mode for the copy operation.
 *         Default is MPMode::NONE.
 *
 * @param in1 Shape/layout reference (mdspan, mdarray, etc.). The output is
 *            allocated to be "like" this input.
 * @param in2 Source data to copy from (mdspan, mdarray, scalar, etc.).
 *
 * @return Newly allocated container (typically mdarray-like) that has the same
 *         shape/layout as `in1` and contains the copied values from `in2`.
 *
 * @note This is equivalent to:
 *       - out = empty_like(in1)
 *       - copy_to(in2, out)
 *       in terms of allocation + broadcasting semantics provided by copy_to.
 *
 * @see mdtensor::empty_like
 * @see mdtensor::copy_to
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto copy_like(in1_t &&in1, in2_t &&in2) {
    auto out = empty_like<dtype>(std::forward<in1_t>(in1));
    copy_to<mpmode>(std::forward<in2_t>(in2), out);
    return out;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/copy_like.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/eye.hpp
/**
 * @file
 * @brief Identity matrix (eye) creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in_t> inline constexpr void eye_impl(in_t &&in) {
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    for (index_t i = 0; i < in.extent(0); i++) {
        for (index_t j = 0; j < in.extent(1); j++) {
            in(i, j) = (i == j) ? 1 : 0;
        }
    }
}

} // namespace detail

/**
 * @brief Fill a 2D tensor with an identity matrix pattern (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Output matrix to fill (mdspan, mdarray, etc.). Rank must be greater
 *        than or equal to 2.
 *
 * @note This function writes `1` on the main diagonal and `0` elsewhere.
 *
 * @see mdtensor::eye for the out-of-place version that allocates and returns
 * an identity matrix.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t>
inline constexpr void eye_to(in_t &&in) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::eye_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2>{}, core::to_mdspan(std::forward<in_t>(in)));
}

/**
 * @brief Create an identity matrix (out-of-place).
 *
 * @tparam dtype Element type of the created matrix.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam exts_t (optional) Extents type describing the matrix shape. Default
 * is extents<uint8_t>.
 *
 * @param exts Output matrix extents (shape). Rank must be greater than or equal
 * to 2.
 *
 * @return A newly allocated matrix filled as an identity matrix.
 *
 * @note Equivalent to allocating `empty<dtype>(exts)` and then calling
 *       `eye_to` on it.
 *
 * @see mdtensor::eye_to for the in-place version that fills an existing output.
 */
template <typename dtype, MPMode mpmode = MPMode::NONE,
          extents_c exts_t = extents<uint8_t>>
[[nodiscard]] inline constexpr auto eye(exts_t &&exts = exts_t{}) {
    auto out = empty<dtype>(std::forward<exts_t>(exts));
    eye_to<mpmode>(out);
    return out;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/eye.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/fill.hpp
/**
 * @file
 * @brief Fill utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in_t, typename val_t>
inline constexpr void fill_impl(in_t &&in, val_t &&val) {
    in() = val;
}

} // namespace detail

/**
 * @brief Fill all elements of a tensor with a scalar value (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Output tensor to fill (mdspan, mdarray, scalar, etc.).
 * @param val Value to assign to every element.
 *
 * @note Equivalent to `in[...] = val` in terms of array broadcasting.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename val_t>
inline constexpr void fill(in_t &&in, val_t &&val) {
    core::batch<mpmode>(
        [&](auto &&...elems) {
            detail::fill_impl(std::forward<decltype(elems)>(elems)..., val);
        },
        core::to_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/fill.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/full.hpp
/**
 * @file
 * @brief Full creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

/**
 * @brief Create a new tensor filled with a scalar value (out-of-place).
 *
 * @tparam dtype Element type of the result tensor.
 * @tparam mpmode (optional) Parallel execution mode used for filling. Default
 * is MPMode::NONE.
 * @tparam exts_t (optional) Extents type. Default is extents<uint8_t>.
 *
 * @param val Fill value.
 * @param exts Output extents.
 *
 * @return Newly allocated tensor (mdarray) with extents `exts`, filled with
 * `val`.
 *
 * @note Equivalent to `out = empty<dtype>(exts); fill(out, val);`.
 *
 * @see mdtensor::empty
 * @see mdtensor::fill
 */
template <typename dtype, MPMode mpmode = MPMode::NONE,
          extents_c exts_t = extents<uint8_t>>
[[nodiscard]] inline constexpr auto full(dtype &&val,
                                         exts_t &&exts = exts_t{}) {
    auto out = empty<dtype>(std::forward<exts_t>(exts));
    fill<mpmode>(out, std::forward<dtype>(val));
    return out;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/full.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/full_like.hpp
/**
 * @file
 * @brief Full-like creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

/**
 * @brief Create a new tensor filled with a scalar value, matching an input's
 *        shape (out-of-place).
 *
 * @tparam dtype (optional) Data type of the result. If void, deduced from the
 * input value type.
 * @tparam mpmode (optional) Parallel execution mode used for filling. Default
 * is MPMode::NONE.
 *
 * @param in Reference tensor whose extents are used (mdspan, mdarray, etc.).
 * @param val Fill value.
 *
 * @return Newly allocated tensor (mdarray) with the same extents as `in`,
 * filled with `val`.
 *
 * @note Equivalent to `out = empty_like(in); out[...] = val`.
 *
 * @see mdtensor::empty_like
 * @see mdtensor::fill
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t,
          typename val_t>
[[nodiscard]] inline constexpr auto full_like(in_t &&in, val_t &&val) {
    auto out = empty_like<dtype>(std::forward<in_t>(in));
    fill<mpmode>(out, std::forward<val_t>(val));
    return out;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/full_like.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/linspace.hpp
/**
 * @file
 * @brief Linspace utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



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
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/linspace.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/ones.hpp
/**
 * @file
 * @brief Ones tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

/**
 * @brief Create a tensor filled with ones (general extents overload).
 *
 * @tparam dtype Output value type.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam exts_t (optional) Extents type specifying the tensor shape.
 * Default is `extents<uint8_t>`.
 *
 * @param exts Extents describing the output shape.
 *
 * @return mdarray filled with ones.
 *
 * @note Equivalent to full(1, exts).
 *
 * @see mdtensor::full for the general fill-value version.
 */
template <typename dtype, MPMode mpmode = MPMode::NONE,
          extents_c exts_t = extents<uint8_t>>
[[nodiscard]] inline constexpr auto ones(exts_t &&exts = exts_t{}) {
    return full<dtype, mpmode, exts_t>(1, std::forward<exts_t>(exts));
}

/**
 * @brief Create a 1D tensor filled with ones (length overload).
 *
 * @tparam dtype Output value type.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param len Length of the output 1D tensor.
 *
 * @return 1D mdarray of length `len`, filled with ones.
 *
 * @note Equivalent to full(1, dims<1>{len}).
 *
 * @see mdtensor::ones(extents) for the general extents version.
 */
template <typename dtype, MPMode mpmode = MPMode::NONE>
[[nodiscard]] inline constexpr auto ones(const size_t &len) {
    return full<dtype, mpmode>(1, mdtensor::dims<1>{len});
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/ones.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/ones_like.hpp
/**
 * @file
 * @brief Ones-like tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

/**
 * @brief Create a tensor filled with ones, matching the shape of the input.
 *
 * @tparam dtype (optional) Output value type. If void, deduced from input.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam in_t Input tensor type (mdspan, mdarray, etc.).
 *
 * @param in Input tensor whose extents determine the output shape.
 *
 * @return mdarray with the same extents as `in`, filled with ones.
 *
 * @note Equivalent to full_like(in, 1).
 *
 * @see mdtensor::full_like for the general fill-value version.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto ones_like(in_t &&in) {
    return full_like<dtype, mpmode>(std::forward<in_t>(in), 1);
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/ones_like.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/zeros.hpp
/**
 * @file
 * @brief Zeros tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

/**
 * @brief Create a tensor filled with zeros (general extents overload).
 *
 * @tparam dtype Output value type.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam exts_t (optional) Extents type specifying the tensor shape.
 * Default is `extents<uint8_t>`.
 *
 * @param exts Extents describing the output shape.
 *
 * @return mdarray filled with zeros.
 *
 * @note Equivalent to full(0, exts).
 *
 * @see mdtensor::full for the general fill-value version.
 */
template <typename dtype, MPMode mpmode = MPMode::NONE,
          extents_c exts_t = extents<uint8_t>>
[[nodiscard]] inline constexpr auto zeros(exts_t &&exts = exts_t{}) {
    return full<dtype, mpmode, exts_t>(0, std::forward<exts_t>(exts));
}

/**
 * @brief Create a 1D tensor filled with zeros (length overload).
 *
 * @tparam dtype Output value type.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param len Length of the output 1D tensor.
 *
 * @return 1D mdarray of length `len`, filled with zeros.
 *
 * @note Equivalent to full(0, dims<1>{len}).
 *
 * @see mdtensor::zeros(extents) for the general extents version.
 */
template <typename dtype, MPMode mpmode = MPMode::NONE>
[[nodiscard]] inline constexpr auto zeros(const size_t &len) {
    return full<dtype, mpmode>(0, mdtensor::dims<1>{len});
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/zeros.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/zeros_like.hpp
/**
 * @file
 * @brief Zeros-like tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

/**
 * @brief Create a tensor filled with zeros, matching the shape of the input.
 *
 * @tparam dtype (optional) Output value type. If void, deduced from input.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam in_t Input tensor type (mdspan, mdarray, etc.).
 *
 * @param in Input tensor whose extents determine the output shape.
 *
 * @return mdarray with the same extents as `in`, filled with zeros.
 *
 * @note Equivalent to full_like(in, 0).
 *
 * @see mdtensor::full_like for the general fill-value version.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto zeros_like(in_t &&in) {
    return full_like<dtype, mpmode>(std::forward<in_t>(in), 0);
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/zeros_like.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/creation.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/linalg.hpp
/**
 * @file
 * @brief Linear algebra module includes for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/inv.hpp
/**
 * @file
 * @brief Matrix inverse utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/absolute.hpp
/**
 * @file
 * @brief Element-wise absolute value utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in_t, typename out_t>
inline constexpr void absolute_impl(in_t &&in, out_t &&out) {
#ifdef REAL_GCC
    out() = std::abs(in());

#else
    // NOTE: std::abs is not constexpr in clang 16.
    if constexpr (std::is_signed_v<std::remove_cvref_t<decltype(in())>>) {
        out() = in() < 0 ? -in() : in();

    } else {
        out() = in();
    }

#endif
}

} // namespace detail

/**
 * @brief Compute absolute value element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = std::abs(in) in terms of array broadcasting.
 *
 * @see mdtensor::absolute for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void absolute_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::absolute_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compute absolute value element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = std::abs(in) in terms of array broadcasting.
 *
 * @see mdtensor::absolute_to for the in-place version that writes into an
 *      output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto absolute(in_t &&in) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::absolute_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/absolute.hpp

namespace mdtensor {
namespace linalg {
namespace detail {

template <md_c in_t, md_c out_t>
[[nodiscard]] inline constexpr bool inv_naive(in_t &&in, out_t &&out) {
    auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    static_assert(decltype(in_mds)::rank() == 2);
    static_assert(decltype(out_mds)::rank() == 2);

    using index_t = typename decltype(in_mds)::index_type;

    const index_t n = in_mds.extent(0);

    if (in_mds.extent(0) != in_mds.extent(1) ||
        in_mds.extent(0) != out_mds.extent(0) ||
        in_mds.extent(0) != out_mds.extent(1)) {
        return false;
    }

    auto in_copy = copy(in_mds);
    eye_to(out_mds);

    for (index_t i = 0; i < n; i++) {
        index_t pivot_row = i;
        auto max_abs = absolute(in_copy(i, i));

        for (index_t row = i + 1; row < n; row++) {
            const auto candidate = absolute(in_copy(row, i));

            if (candidate > max_abs) {
                max_abs = candidate;
                pivot_row = row;
            }
        }

        if (max_abs == 0) {
            // Handle error: singular matrix (no inverse)
            return false;
        }

        if (pivot_row != i) {
            for (index_t j = 0; j < n; j++) {
                std::swap(in_copy(i, j), in_copy(pivot_row, j));
                std::swap(out_mds(i, j), out_mds(pivot_row, j));
            }
        }

        const auto pivot = in_copy(i, i);

        for (index_t j = 0; j < n; j++) {
            in_copy(i, j) /= pivot;
            out_mds(i, j) /= pivot;
        }

        for (index_t j = 0; j < n; j++) {
            if (i == j) {
                continue;
            }

            const auto factor = in_copy(j, i);

            if (factor == 0) {
                continue;
            }

            for (index_t k = 0; k < n; k++) {
                in_copy(j, k) -= factor * in_copy(i, k);
                out_mds(j, k) -= factor * out_mds(i, k);
            }
        }
    }

    return true;
}

template <md_c in_t, md_c out_t>
[[nodiscard]] inline constexpr bool inv_impl(in_t &&in, out_t &&out) {
    static_assert(std::remove_cvref_t<in_t>::rank() == 2);
    static_assert(std::remove_cvref_t<out_t>::rank() == 2);

#ifdef MDTENSOR_USE_EIGEN
#if __cplusplus >= 202302L // TODO: Impliement for C++20
    if constexpr (core::eigen::eigen_mappable_mdspan_c<in_t> &&
                  core::eigen::eigen_mappable_mdspan_c<out_t>) {
        if (!std::is_constant_evaluated()) [[likely]] {
            const auto ein = core::eigen::to_eigen(in);
            auto eout = core::eigen::to_eigen(out);

            eout = ein.inverse();

            return true;
        }
    }

#else
    assert(false && "Eigen inverse not implemented for C++20");

#endif
#endif

    return inv_naive(in, out);
}

} // namespace detail

/**
 * @brief Compute matrix inverse (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input matrix (mdspan, mdarray, etc.) (... x N x N).
 * @param out Output matrix (mdspan, mdarray, etc.) (... x N x N).
 * @param valid Output mdspan, mdarray, or scalar indicating validity of the
 * inverse.
 *
 * @note The inverse is computed per-matrix when `in` is batched. The `valid`
 *       flag is produced per matrix instance.
 *
 * @see mdtensor::linalg::inv for the out-of-place version that returns the
 * result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t,
          typename valid_t>
inline constexpr void inv_to(in_t &&in, out_t &&out, valid_t &&valid) {
    core::batch<mpmode>(
        [](auto &&in, auto &&out, auto &&valid) {
            valid() = detail::inv_impl(std::forward<decltype(in)>(in),
                                       std::forward<decltype(out)>(out));
        },
        std::index_sequence<2, 2, 0>{},
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<out_t>(out)),
        core::to_mdspan(std::forward<valid_t>(valid)));
}

/**
 * @brief Compute matrix inverse (out-of-place).
 *
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 * input.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input matrix (mdspan, mdarray, etc.) (... x N x N).
 *
 * @return A tuple-like output from batch_out consisting of:
 *         - inverse matrix (mdarray) (... x N x N).
 *         - validity flag (mdarray or scalar) (...), indicating if the inverse
 *        was successfully computed for each matrix instance.
 *
 * @see mdtensor::linalg::inv_to for the in-place version that writes into an
 * output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto inv(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    auto out = core::create_out<dtype>(
        std::index_sequence<2>{}, core::slice_from_right<2>(in_mds.extents()),
        in_mds);
    auto valid = core::create_out<uint8_t>(std::index_sequence<2>{},
                                           extents<uint8_t>{}, in_mds);

    inv_to<mpmode>(in_mds, out, valid);

    return std::pair{out, valid};
}

} // namespace linalg
} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/inv.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/lu.hpp
/**
 * @file
 * @brief SciPy-like LU decomposition utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace linalg {
namespace detail {

template <md_c in_t, md_c p_indices_t, md_c l_t, md_c u_t>
inline constexpr void lu_p_indices_impl(in_t &&in, p_indices_t &&p_indices,
                                        l_t &&l, u_t &&u) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto p_indices_mds =
        core::to_mdspan(std::forward<p_indices_t>(p_indices));
    const auto l_mds = core::to_mdspan(std::forward<l_t>(l));
    const auto u_mds = core::to_mdspan(std::forward<u_t>(u));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using p_indices_mds_t = std::remove_cvref_t<decltype(p_indices_mds)>;
    using l_mds_t = std::remove_cvref_t<decltype(l_mds)>;
    using u_mds_t = std::remove_cvref_t<decltype(u_mds)>;

    static_assert(in_mds_t::rank() == 2);
    static_assert(p_indices_mds_t::rank() == 1);
    static_assert(l_mds_t::rank() == 2);
    static_assert(u_mds_t::rank() == 2);

    using index_t = typename in_mds_t::index_type;

    constexpr size_t m_s = in_mds_t::static_extent(0);

    const index_t m = in_mds.extent(0);
    const index_t n = in_mds.extent(1);
    const index_t k = m < n ? m : n;

    assert(p_indices_mds.extent(0) == m);
    assert(l_mds.extent(0) == m);
    assert(l_mds.extent(1) == k);
    assert(u_mds.extent(0) == k);
    assert(u_mds.extent(1) == n);

    // initialize
    auto in_copy = copy(in_mds);
    auto row_order = core::create_data<index_t>(extents<index_t, m_s>{m});
    for (index_t i = 0; i < m; i++) {
        row_order(i) = i;
    }

    // Compute LU decomposition with partial pivoting
    for (index_t i = 0; i < k; i++) {
        // find maximum element in the current column
        index_t pivot_row = i;
        auto max_abs = absolute(in_copy(i, i));

        for (index_t j = i + 1; j < m; j++) {
            const auto candidate = absolute(in_copy(j, i));
            if (candidate > max_abs) {
                max_abs = candidate;
                pivot_row = j;
            }
        }

        // swap row if necessary
        if (pivot_row != i) {
            for (index_t j = 0; j < n; j++) {
                std::swap(in_copy(i, j), in_copy(pivot_row, j));
            }
            std::swap(row_order(i), row_order(pivot_row));
        }

        // no multiplier can be formed
        if (max_abs == 0) {
            continue;
        }

        // compute the multipliers and update the U matrix
        for (index_t j = i + 1; j < m; j++) {
            in_copy(j, i) /= in_copy(i, i);

            const auto factor = in_copy(j, i);

            for (index_t k = i + 1; k < n; k++) {
                in_copy(j, k) -= factor * in_copy(i, k);
            }
        }
    }

    // Generate P
    for (index_t i = 0; i < m; i++) {
        p_indices_mds(row_order(i)) =
            static_cast<typename p_indices_mds_t::value_type>(i);
    }

    // Generate L
    for (index_t i = 0; i < m; i++) {
        for (index_t j = 0; j < k; j++) {
            if (i > j) {
                l_mds(i, j) = in_copy(i, j);

            } else if (i == j) {
                l_mds(i, j) = 1;

            } else {
                l_mds(i, j) = 0;
            }
        }
    }

    // Generate U
    for (index_t i = 0; i < k; i++) {
        for (index_t j = 0; j < n; j++) {
            if (i <= j) {
                u_mds(i, j) = in_copy(i, j);

            } else {
                u_mds(i, j) = 0;
            }
        }
    }
}

template <md_c in_t, md_c p_t, md_c l_t, md_c u_t>
inline constexpr void lu_full_impl(in_t &&in, p_t &&p, l_t &&l, u_t &&u) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto p_mds = core::to_mdspan(std::forward<p_t>(p));
    const auto l_mds = core::to_mdspan(std::forward<l_t>(l));
    const auto u_mds = core::to_mdspan(std::forward<u_t>(u));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using p_mds_t = std::remove_cvref_t<decltype(p_mds)>;
    using l_mds_t = std::remove_cvref_t<decltype(l_mds)>;
    using u_mds_t = std::remove_cvref_t<decltype(u_mds)>;

    static_assert(in_mds_t::rank() == 2);
    static_assert(p_mds_t::rank() == 2);
    static_assert(l_mds_t::rank() == 2);
    static_assert(u_mds_t::rank() == 2);

    using index_t = typename p_mds_t::index_type;

    constexpr size_t m_s = in_mds_t::static_extent(0);

    const index_t m = in_mds.extent(0);
    const index_t n = in_mds.extent(1);

    auto p_indices = core::create_data<index_t>(extents<index_t, m_s>{m});

    lu_p_indices_impl(in_mds, p_indices, l_mds, u_mds);

    for (index_t i = 0; i < m; i++) {
        for (index_t j = 0; j < m; j++) {
            p_mds(i, j) = (j == p_indices(i) ? 1 : 0);
        }
    }
}

template <md_c in_t, md_c pl_t, md_c u_t>
inline constexpr void lu_permute_l_impl(in_t &&in, pl_t &&pl, u_t &&u) {
    auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    auto pl_mds = core::to_mdspan(std::forward<pl_t>(pl));
    auto u_mds = core::to_mdspan(std::forward<u_t>(u));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using pl_mds_t = std::remove_cvref_t<decltype(pl_mds)>;
    using u_mds_t = std::remove_cvref_t<decltype(u_mds)>;

    static_assert(in_mds_t::rank() == 2);
    static_assert(pl_mds_t::rank() == 2);
    static_assert(u_mds_t::rank() == 2);

    using index_t = typename in_mds_t::index_type;
    using value_t = typename pl_mds_t::value_type;

    constexpr size_t m_s = in_mds_t::static_extent(0);
    constexpr size_t n_s = in_mds_t::static_extent(1);
    constexpr size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(0);
    const index_t n = in_mds.extent(1);
    const index_t k = m < n ? m : n;

    auto p_indices = core::create_data<index_t>(extents<index_t, m_s>{m});
    auto l = core::create_data<value_t>(extents<index_t, m_s, k_s>{m, k});

    lu_p_indices_impl(in_mds, p_indices, l, u_mds);

    // Apply the permutation to L
    for (index_t i = 0; i < m; i++) {
        for (index_t j = 0; j < k; j++) {
            pl_mds(i, j) = l(p_indices(i), j);
        }
    }
}

} // namespace detail

template <MPMode mpmode = MPMode::NONE, typename in_t, typename p_indices_t,
          typename l_t, typename u_t>
inline constexpr void lu_p_indices_to(in_t &&in, p_indices_t &&p_indices,
                                      l_t &&l, u_t &&u) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::lu_p_indices_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 1, 2, 2>{},
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<p_indices_t>(p_indices)),
        core::to_mdspan(std::forward<l_t>(l)),
        core::to_mdspan(std::forward<u_t>(u)));
}

template <MPMode mpmode = MPMode::NONE, typename in_t, typename p_t,
          typename l_t, typename u_t>
inline constexpr void lu_full_to(in_t &&in, p_t &&p, l_t &&l, u_t &&u) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::lu_full_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 2, 2, 2>{},
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<p_t>(p)),
        core::to_mdspan(std::forward<l_t>(l)),
        core::to_mdspan(std::forward<u_t>(u)));
}

template <MPMode mpmode = MPMode::NONE, typename in_t, typename pl_t,
          typename u_t>
inline constexpr void lu_permute_l_to(in_t &&in, pl_t &&pl, u_t &&u) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::lu_permute_l_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 2, 2>{},
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<pl_t>(pl)),
        core::to_mdspan(std::forward<u_t>(u)));
}

template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto lu_p_indices(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;

    using index_t = typename in_mds_t::index_type;

    constexpr size_t rank = in_mds_t::rank();

    constexpr size_t m_s = in_mds_t::static_extent(rank - 2);
    constexpr size_t n_s = in_mds_t::static_extent(rank - 1);
    constexpr size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(rank - 2);
    const index_t n = in_mds.extent(rank - 1);
    const index_t k = m < n ? m : n;

    auto p_indices = core::create_out<index_t>(
        std::index_sequence<2>{}, extents<index_t, m_s>{m}, in_mds);
    auto l = core::create_out<dtype>(std::index_sequence<2>{},
                                     extents<index_t, m_s, k_s>{m, k}, in_mds);
    auto u = core::create_out<dtype>(std::index_sequence<2>{},
                                     extents<index_t, k_s, n_s>{k, n}, in_mds);

    lu_p_indices_to<mpmode>(in_mds, p_indices, l, u);

    return std::tuple{p_indices, l, u};
}

template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto lu_full(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;

    using index_t = typename in_mds_t::index_type;

    constexpr size_t rank = in_mds_t::rank();

    constexpr size_t m_s = in_mds_t::static_extent(rank - 2);
    constexpr size_t n_s = in_mds_t::static_extent(rank - 1);
    constexpr size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(rank - 2);
    const index_t n = in_mds.extent(rank - 1);
    const index_t k = m < n ? m : n;

    auto p = core::create_out<index_t>(
        std::index_sequence<2>{}, extents<index_t, m_s, m_s>{m, m}, in_mds);
    auto l = core::create_out<dtype>(std::index_sequence<2>{},
                                     extents<index_t, m_s, k_s>{m, k}, in_mds);
    auto u = core::create_out<dtype>(std::index_sequence<2>{},
                                     extents<index_t, k_s, n_s>{k, n}, in_mds);

    lu_full_to<mpmode>(in_mds, p, l, u);

    return std::tuple{p, l, u};
}

template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto lu_permute_l(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;

    using index_t = typename in_mds_t::index_type;

    constexpr size_t rank = in_mds_t::rank();

    constexpr size_t m_s = in_mds_t::static_extent(rank - 2);
    constexpr size_t n_s = in_mds_t::static_extent(rank - 1);
    constexpr size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(rank - 2);
    const index_t n = in_mds.extent(rank - 1);
    const index_t k = m < n ? m : n;

    auto pl = core::create_out<dtype>(std::index_sequence<2>{},
                                      extents<index_t, m_s, k_s>{m, k}, in_mds);
    auto u = core::create_out<dtype>(std::index_sequence<2>{},
                                     extents<index_t, k_s, n_s>{k, n}, in_mds);

    lu_permute_l_to<mpmode>(in_mds, pl, u);

    return std::pair{pl, u};
}

template <bool permute_l = false, bool p_indices = false, typename dtype = void,
          MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto lu(in_t &&in) {
    static_assert(!(permute_l && p_indices),
                  "lu cannot return both permuted L and P indices.");

    if constexpr (permute_l) {
        return lu_permute_l<dtype, mpmode>(std::forward<in_t>(in));

    } else if constexpr (p_indices) {
        return lu_p_indices<dtype, mpmode>(std::forward<in_t>(in));

    } else {
        return lu_full<dtype, mpmode>(std::forward<in_t>(in));
    }
}

} // namespace linalg
} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/lu.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/matmul.hpp
/**
 * @file
 * @brief Matrix-matrix multiplication utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace linalg {
namespace detail {

template <md_c in1_t, md_c in2_t, md_c out_t>
inline constexpr void matmul_naive_noalias(in1_t &&in1, in2_t &&in2,
                                           out_t &&out) noexcept {
    using out_index_t = typename std::remove_cvref_t<out_t>::index_type;
    using in1_index_t = typename std::remove_cvref_t<in1_t>::index_type;

    for (out_index_t i = 0; i < out.extent(0); i++) {
        for (out_index_t j = 0; j < out.extent(1); j++) {
            out(i, j) = 0;

            for (in1_index_t k = 0; k < in1.extent(1); k++) {
                out(i, j) += in1(i, k) * in2(k, j);
            }
        }
    }
}

template <md_c in1_t, md_c in2_t, md_c out_t>
inline constexpr void matmul_naive(in1_t &&in1, in2_t &&in2,
                                   out_t &&out) noexcept {
    const auto in1_mds = core::to_const_mdspan(std::forward<in1_t>(in1));
    const auto in2_mds = core::to_const_mdspan(std::forward<in2_t>(in2));
    auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    if (std::is_constant_evaluated()) {
        auto out_tmp = empty_like(out_mds);
        matmul_naive_noalias(in1_mds, in2_mds, out_tmp.to_mdspan());
        copy_to(out_tmp, out_mds);
        return;
    }

    bool need_copy = false;

    if constexpr (requires {
                      in1_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in1_mds.data_handle() == out_mds.data_handle()) [[unlikely]] {
            need_copy = true;
        }
    }

    if constexpr (requires {
                      in2_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in2_mds.data_handle() == out_mds.data_handle()) [[unlikely]] {
            need_copy = true;
        }
    }

    if (!need_copy) [[likely]] {
        matmul_naive_noalias(in1_mds, in2_mds, out_mds);

    } else [[unlikely]] {
        auto out_tmp = empty_like(out_mds);
        matmul_naive_noalias(in1_mds, in2_mds, out_tmp.to_mdspan());
        copy_to(out_tmp, out_mds);
    }
}

template <md_c in1_t, md_c in2_t, md_c out_t>
inline constexpr void matmul_impl(in1_t &&in1, in2_t &&in2,
                                  out_t &&out) noexcept {
    static_assert(std::remove_cvref_t<in1_t>::rank() == 2);
    static_assert(std::remove_cvref_t<in2_t>::rank() == 2);
    static_assert(std::remove_cvref_t<out_t>::rank() == 2);

#ifdef MDTENSOR_USE_EIGEN
#if __cplusplus >= 202302L // TODO: Impliement for C++20
    if constexpr (core::eigen::eigen_mappable_mdspan_c<in1_t> &&
                  core::eigen::eigen_mappable_mdspan_c<in2_t> &&
                  core::eigen::eigen_mappable_mdspan_c<out_t>) {
        if (!std::is_constant_evaluated() && 8 <= out.extent(0) + out.extent(1))
            [[likely]] {
            using value_t = core::common_data_type_t<
                typename std::remove_cvref_t<in1_t>::value_type,
                typename std::remove_cvref_t<in2_t>::value_type>;

            const auto ein1 =
                core::eigen::to_eigen(in1).template cast<value_t>();
            const auto ein2 =
                core::eigen::to_eigen(in2).template cast<value_t>();
            auto eout = core::eigen::to_eigen(out);

            eout = (ein1 * ein2)
                       .template cast<
                           typename std::remove_cvref_t<out_t>::value_type>();

            return;
        }
    }

#else
    assert(false && "Eigen inverse not implemented for C++20");

#endif
#endif

    matmul_naive(in1, in2, out);
}

} // namespace detail

/**
 * @brief Matrix multiplication (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input matrix (mdspan, mdarray, etc.) (... x M x K).
 * @param in2 Second input matrix (mdspan, mdarray, etc.) (... x K x N).
 * @param out Output matrix (mdspan, mdarray, etc.) (... x M x N).
 *
 * @note The multiplication is performed per-matrix when inputs are batched.
 *
 * @see mdtensor::linalg::matmul for the out-of-place version that returns the
 * result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void matmul_to(in1_t &&in1, in2_t &&in2,
                                out_t &&out) noexcept {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::matmul_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 2, 2>{},
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Matrix multiplication (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 * input.
 *
 * @param in1 First input matrix (mdspan, mdarray, etc.) (... x M x K).
 * @param in2 Second input matrix (mdspan, mdarray, etc.) (... x K x N).
 *
 * @return matrix (mdarray) matching the batch extents of the inputs.
 *
 * @note The multiplication is performed per-matrix when inputs are batched.
 *
 * @see mdtensor::linalg::matmul_to for the in-place version that writes into an
 * output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto matmul(in1_t &&in1, in2_t &&in2) noexcept {
    const auto in1_mds = core::to_const_mdspan(std::forward<in1_t>(in1));
    const auto in2_mds = core::to_const_mdspan(std::forward<in2_t>(in2));

    const auto uin1_exts = core::slice_from_right<2>(in1_mds.extents());
    const auto uin2_exts = core::slice_from_right<2>(in2_mds.extents());
    const auto uout_exts = extents<
        core::common_index_type_t<typename decltype(uin1_exts)::index_type,
                                  typename decltype(uin2_exts)::index_type>,
        decltype(uin1_exts)::static_extent(0),
        decltype(uin2_exts)::static_extent(1)>{uin1_exts.extent(0),
                                               uin2_exts.extent(1)};

    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::matmul_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 2>{}, uout_exts, in1_mds, in2_mds);
}

} // namespace linalg

inline constexpr void matmul_to(auto &&...elems) noexcept {
    linalg::matmul_to(std::forward<decltype(elems)>(elems)...);
}

template <typename dtype = void>
[[nodiscard]] inline constexpr auto matmul(auto &&...elems) noexcept {
    return linalg::matmul<dtype>(std::forward<decltype(elems)>(elems)...);
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/matmul.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/matvec.hpp
/**
 * @file
 * @brief Matrix-vector multiplication utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace linalg {
namespace detail {

template <md_c in1_t, md_c in2_t, md_c out_t>
inline constexpr void matvec_naive_noalias(in1_t &&in1, in2_t &&in2,
                                           out_t &&out) noexcept {
    using out_index_t = typename std::remove_cvref_t<out_t>::index_type;
    using in1_index_t = typename std::remove_cvref_t<in1_t>::index_type;

    for (out_index_t i = 0; i < out.extent(0); i++) {
        out(i) = 0;

        for (in1_index_t j = 0; j < in1.extent(1); j++) {
            out(i) += in1(i, j) * in2(j);
        }
    }
}

template <md_c in1_t, md_c in2_t, md_c out_t>
inline constexpr void matvec_naive(in1_t &&in1, in2_t &&in2,
                                   out_t &&out) noexcept {
    const auto in1_mds = core::to_const_mdspan(std::forward<in1_t>(in1));
    const auto in2_mds = core::to_const_mdspan(std::forward<in2_t>(in2));
    auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    if (std::is_constant_evaluated()) {
        auto out_tmp = empty_like(out_mds);
        matvec_naive_noalias(in1_mds, in2_mds, out_tmp.to_mdspan());
        copy_to(out_tmp, out_mds);
        return;
    }

    bool need_copy = false;

    if constexpr (requires {
                      in1_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in1_mds.data_handle() == out_mds.data_handle()) [[unlikely]] {
            need_copy = true;
        }
    }

    if constexpr (requires {
                      in2_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in2_mds.data_handle() == out_mds.data_handle()) [[unlikely]] {
            need_copy = true;
        }
    }

    if (!need_copy) [[likely]] {
        matvec_naive_noalias(in1_mds, in2_mds, out_mds);

    } else [[unlikely]] {
        auto out_tmp = empty_like(out_mds);
        matvec_naive_noalias(in1_mds, in2_mds, out_tmp.to_mdspan());
        copy_to(out_tmp, out_mds);
    }
}

template <md_c in1_t, md_c in2_t, md_c out_t>
inline constexpr void matvec_impl(in1_t &&in1, in2_t &&in2,
                                  out_t &&out) noexcept {
#ifdef MDTENSOR_USE_EIGEN
#if __cplusplus >= 202302L // TODO: Impliement for C++20
    if constexpr (core::eigen::eigen_mappable_mdspan_c<in1_t> &&
                  core::eigen::eigen_mappable_mdspan_c<in2_t> &&
                  core::eigen::eigen_mappable_mdspan_c<out_t>) {
        if (!std::is_constant_evaluated() && 8 <= out.extent(0) + out.extent(1))
            [[likely]] {
            using value_t = core::common_data_type_t<
                typename std::remove_cvref_t<in1_t>::value_type,
                typename std::remove_cvref_t<in2_t>::value_type>;

            const auto ein1 =
                core::eigen::to_eigen(in1).template cast<value_t>();
            const auto ein2 =
                core::eigen::to_eigen(in2).template cast<value_t>();
            auto eout = core::eigen::to_eigen(out);

            eout = (ein1 * ein2)
                       .template cast<
                           typename std::remove_cvref_t<out_t>::value_type>();

            return;
        }
    }

#else
    assert(false && "Eigen inverse not implemented for C++20");

#endif
#endif

    matvec_naive(in1, in2, out);
}

} // namespace detail

/**
 * @brief Matrix-vector multiplication (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 Input matrix (mdspan, mdarray, etc.) (... x M x K).
 * @param in2 Input vector (mdspan, mdarray, etc.) (... x K).
 * @param out Output vector (mdspan, mdarray, etc.) (... x M).
 *
 * @note The multiplication is performed per instance when inputs are batched.
 *       The last 2 axes of `in1` are treated as matrix axes, and the last axis
 *       of `in2` / `out` is treated as the vector axis.
 *
 * @see mdtensor::linalg::matvec for the out-of-place version that returns the
 * result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void matvec_to(in1_t &&in1, in2_t &&in2,
                                out_t &&out) noexcept {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::matvec_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 1, 1>{},
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Matrix-vector multiplication (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 * inputs.
 *
 * @param in1 Input matrix (mdspan, mdarray, etc.) (... x M x K).
 * @param in2 Input vector (mdspan, mdarray, etc.) (... x K).
 *
 * @return Vector (mdarray) with shape (... x M).
 *
 * @note The multiplication is performed per instance when inputs are batched.
 *
 * @see mdtensor::linalg::matvec_to for the in-place version that writes into an
 * output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto matvec(in1_t &&in1, in2_t &&in2) noexcept {
    const auto in1_mds = core::to_const_mdspan(std::forward<in1_t>(in1));
    const auto in2_mds = core::to_const_mdspan(std::forward<in2_t>(in2));

    const auto uin1_exts = core::slice_from_right<2>(in1_mds.extents());
    const auto uin2_exts = core::slice_from_right<2>(in2_mds.extents());
    const auto uout_exts = extents<
        core::common_index_type_t<typename decltype(uin1_exts)::index_type,
                                  typename decltype(uin2_exts)::index_type>,
        decltype(uin1_exts)::static_extent(0)>{uin1_exts.extent(0)};

    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::matvec_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 1>{}, uout_exts, in1_mds, in2_mds);
}

} // namespace linalg

inline constexpr void matvec_to(auto &&...elems) noexcept {
    linalg::matvec_to(std::forward<decltype(elems)>(elems)...);
}

template <typename dtype = void>
[[nodiscard]] inline constexpr auto matvec(auto &&...elems) noexcept {
    return linalg::matvec<dtype>(std::forward<decltype(elems)>(elems)...);
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/matvec.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/norm.hpp
/**
 * @file
 * @brief Vector norm (L2) utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/multiply.hpp
/**
 * @file
 * @brief Element-wise multiplication utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void multiply_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    out() = in1() * in2();
}

} // namespace detail

/**
 * @brief Multiply arguments element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = in1 * in2 in terms of array broadcasting.
 *
 * @see mdtensor::multiply for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void multiply_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::multiply_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Multiply arguments element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         inputs.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = in1 * in2 in terms of array broadcasting.
 *
 * @see mdtensor::multiply_to for the in-place version that writes into an
 *      output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto multiply(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::multiply_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/multiply.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/sqrt.hpp
/**
 * @file
 * @brief Element-wise square root utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

#ifndef REAL_GCC

template <std::floating_point dtype>
[[nodiscard]] inline constexpr dtype
sqrt_newton_raphson(dtype &&x, dtype &&curr, dtype &&prev) {
    return (curr == prev)
               ? curr
               : sqrt_newton_raphson(x, (curr + x / curr) / (dtype)2, curr);
}

#endif

template <typename in_t, typename out_t>
inline constexpr void sqrt_impl(in_t &&in, out_t &&out) {
#ifdef REAL_GCC
    out() = std::sqrt(in());

#else
    using value_t = core::common_data_type_t<decltype(in()), float>;

    out() = (in() >= 0 && in() < std::numeric_limits<value_t>::infinity())
                ? sqrt_newton_raphson(static_cast<value_t>(in()),
                                      static_cast<value_t>(in()),
                                      static_cast<value_t>(0))
                : std::numeric_limits<value_t>::quiet_NaN();

#endif
}

} // namespace detail

/**
 * @brief Compute square root element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = std::sqrt(in) in terms of array broadcasting.
 *
 * @see mdtensor::sqrt for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void sqrt_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::sqrt_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compute square root element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = std::sqrt(in) in terms of array broadcasting.
 *
 * @see mdtensor::sqrt_to for the in-place version that writes into an output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto sqrt(in_t &&in) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::sqrt_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/sqrt.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/sum.hpp
/**
 * @file
 * @brief Sum reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/add.hpp
/**
 * @file
 * @brief Element-wise addition utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void add_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    out() = in1() + in2();
}

} // namespace detail

/**
 * @brief Add arguments element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = in1 + in2 in terms of array broadcasting.
 *
 * @see mdtensor::add for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void add_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::add_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Add arguments element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) data type of the result. If void, deduced from
 *         inputs.
 * @param in1 mdspan, mdarray, scalar, etc.
 * @param in2 mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = in1 + in2 in terms of array broadcasting.
 *
 * @see mdtensor::add_to for the in-place version that modifies the output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto add(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::add_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/add.hpp

namespace mdtensor {
namespace detail {

template <typename in_t, typename out_t>
inline constexpr void sum_impl(in_t &&in, out_t &&out) {
    fill(std::forward<out_t>(out), 0);

    for (typename std::remove_cvref_t<in_t>::index_type i = 0; i < in.extent(0);
         i++) {
        add_to(std::forward<out_t>(out),
               core::submdspan_from_left(std::forward<in_t>(in), i),
               std::forward<out_t>(out));
    }
}

} // namespace detail

/**
 * @brief Compute sum along a specified axis (in-place).
 *
 * @tparam Axis Axis to reduce. Negative values are supported and normalized
 *         by the input rank (NumPy-like semantics).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note The reduction is performed along the specified axis and written to out.
 * @note Broadcasting is not performed; this is a reduction operation.
 *
 * @see mdtensor::sum for the out-of-place axis-reduction version that returns
 *      the result.
 */
template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename in_t,
          typename out_t>
inline constexpr void sum_to(in_t &&in, out_t &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::sum_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank, rin_rank - 1>{}, in_mds, out_mds);
}

/**
 * @brief Compute sum along a specified axis (out-of-place).
 *
 * @tparam Axis Axis to reduce. Negative values are supported and normalized
 *         by the input rank (NumPy-like semantics).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar (depending on the input rank and reduction).
 *
 * @note The reduction is performed along the specified axis.
 *
 * @see mdtensor::sum_to for the in-place version that writes into an output.
 */
template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename dtype = void,
          typename in_t>
[[nodiscard]] inline constexpr auto sum(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::sum_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank>{},
        core::slice_from_right<rin_rank - 1>(in_mds.extents()), in_mds);
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/sum.hpp

namespace mdtensor {
namespace linalg {
namespace detail {

template <md_c in_t, md_c out_t>
inline constexpr void norm_impl(in_t &&in, out_t &&out) noexcept {
    static_assert(std::remove_cvref_t<in_t>::rank() == 1);
    static_assert(std::remove_cvref_t<out_t>::rank() == 0);

    using index_t = typename std::remove_cvref_t<in_t>::index_type;

    out() = 0;
    for (index_t i = 0; i < in.extent(0); i++) {
        out() += in(i)*in(i);
    }

    if (out() > 0) {
        out() = sqrt(out());
    }
}

} // namespace detail

/**
 * @brief Compute Euclidean (L2) norm of a vector (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input tensor (mdspan, mdarray, etc.) (... x N).
 * @param out Output tensor (mdspan, mdarray, scalar, etc.) (...).
 *
 * @note When `mpmode == MPMode::SIMD`, the implementation uses the element-wise
 *       pipeline `sum(multiply(in, in))` followed by `sqrt` to improve
 *       vectorization opportunities.
 *
 * @see mdtensor::linalg::norm for the out-of-place version that returns the
 * result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void norm_to(in_t &&in, out_t &&out) noexcept {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    if constexpr (mpmode == MPMode::SIMD) [[unlikely]] {
        sum_to<-1, mpmode>(multiply<void, mpmode>(in_mds, in_mds), out_mds);
        sqrt_to<mpmode>(out_mds, out_mds);

    } else {
        core::batch<mpmode>(
            [](auto &&...elems) {
                detail::norm_impl(std::forward<decltype(elems)>(elems)...);
            },
            std::index_sequence<1, 0>{}, in_mds, out_mds);
    }
}

/**
 * @brief Compute Euclidean (L2) norm of a vector (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 * input.
 *
 * @param in Input tensor (mdspan, mdarray, etc.) (... x N).
 *
 * @return mdarray or scalar matching the batch extents of the input.
 *
 * @note This function creates a rank-0 output container and then calls
 *       mdtensor::linalg::norm_to.
 *
 * @see mdtensor::linalg::norm_to for the in-place version that writes into an
 * output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto norm(in_t &&in) noexcept {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    auto out = core::create_out<dtype>(std::index_sequence<1>{},
                                       extents<uint8_t>{}, in_mds);

    norm_to<mpmode>(in_mds, out);

    return out;
}

} // namespace linalg
} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/norm.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/solve.hpp
/**
 * @file
 * @brief Linear system solve utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace linalg {
namespace detail {

template <md_c a_t, md_c b_t, md_c x_t>
[[nodiscard]] inline constexpr bool solve_impl(a_t &&a, b_t &&b,
                                               x_t &&x) noexcept {
    auto a_mds = core::to_const_mdspan(std::forward<a_t>(a));
    auto b_mds = core::to_const_mdspan(std::forward<b_t>(b));
    auto x_mds = core::to_mdspan(std::forward<x_t>(x));

    using a_mds_t = std::remove_cvref_t<decltype(a_mds)>;
    using b_mds_t = std::remove_cvref_t<decltype(b_mds)>;
    using x_mds_t = std::remove_cvref_t<decltype(x_mds)>;

    static_assert(a_mds_t::rank() == 2);
    static_assert(b_mds_t::rank() == 1 || b_mds_t::rank() == 2);
    static_assert(x_mds_t::rank() == b_mds_t::rank());

    using index_t = typename a_mds_t::index_type;

    const index_t n = a_mds.extent(0);

    assert(a_mds.extent(1) == n);
    assert(b_mds.extent(0) == n);
    assert(x_mds.extent(0) == n);

    // LU decomposition of A
    const auto [p_indices, l, u] = lu_p_indices(a_mds);

    // check singularity
    for (index_t idx = 0; idx < n; idx++) {
        if (u(idx, idx) == 0) {
            return false;
        }
    }

    if constexpr (b_mds_t::rank() == 1) {
        // initialize out
        for (index_t idx = 0; idx < n; idx++) {
            x_mds(p_indices(idx)) = b_mds(idx);
        }

        // forward substitution
        for (index_t idx = 0; idx < n; idx++) {
            for (index_t jdx = 0; jdx < idx; jdx++) {
                x_mds(idx) -= l(idx, jdx) * x_mds(jdx);
            }
        }

        // backward substitution
        for (index_t i = n; i > 0; i--) {
            const index_t idx = i - 1;

            for (index_t jdx = i; jdx < n; jdx++) {
                x_mds(idx) -= u(idx, jdx) * x_mds(jdx);
            }

            if (u(idx, idx) == 0) {
                return false;
            }

            x_mds(idx) /= u(idx, idx);
        }

    } else {
        const index_t nrhs = b_mds.extent(1);

        assert(x_mds.extent(1) == nrhs);

        for (index_t rhs = 0; rhs < nrhs; rhs++) {
            // initialize out
            for (index_t idx = 0; idx < n; idx++) {
                x_mds(p_indices(idx), rhs) = b_mds(idx, rhs);
            }

            // forward substitution
            for (index_t idx = 0; idx < n; idx++) {
                for (index_t jdx = 0; jdx < idx; jdx++) {
                    x_mds(idx, rhs) -= l(idx, jdx) * x_mds(jdx, rhs);
                }
            }

            // backward substitution
            for (index_t i = n; i > 0; i--) {
                const index_t idx = i - 1;

                for (index_t jdx = i; jdx < n; jdx++) {
                    x_mds(idx, rhs) -= u(idx, jdx) * x_mds(jdx, rhs);
                }

                if (u(idx, idx) == 0) {
                    return false;
                }

                x_mds(idx, rhs) /= u(idx, idx);
            }
        }
    }

    return true;
}

} // namespace detail

template <MPMode mpmode = MPMode::NONE, typename a_t, typename b_t,
          typename x_t, typename valid_t>
inline constexpr void solve_to(a_t &&a, b_t &&b, x_t &&x, valid_t &&valid) {
    const auto a_mds = core::to_const_mdspan(std::forward<a_t>(a));
    const auto b_mds = core::to_const_mdspan(std::forward<b_t>(b));

    constexpr size_t rhs_rank = decltype(b_mds)::rank() == 1 ? 1 : 2;

    core::batch<mpmode>(
        [](auto &&a, auto &&b, auto &&x, auto &&valid) {
            valid() = detail::solve_impl(std::forward<decltype(a)>(a),
                                         std::forward<decltype(b)>(b),
                                         std::forward<decltype(x)>(x));
        },
        std::index_sequence<2, rhs_rank, rhs_rank, 0>{}, a_mds, b_mds,
        core::to_mdspan(std::forward<x_t>(x)),
        core::to_mdspan(std::forward<valid_t>(valid)));
}

template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename a_t,
          typename b_t>
[[nodiscard]] inline constexpr auto solve(a_t &&a, b_t &&b) {
    const auto a_mds = core::to_const_mdspan(std::forward<a_t>(a));
    const auto b_mds = core::to_const_mdspan(std::forward<b_t>(b));

    constexpr size_t rhs_rank = decltype(b_mds)::rank() == 1 ? 1 : 2;

    auto x = core::create_out<dtype>(
        std::index_sequence<2, rhs_rank>{},
        core::slice_from_right<rhs_rank>(b_mds.extents()), a_mds, b_mds);

    auto valid = core::create_out<uint8_t>(std::index_sequence<2, rhs_rank>{},
                                           extents<uint8_t>{}, a_mds, b_mds);

    solve_to<mpmode>(a_mds, b_mds, x, valid);

    return std::pair{x, valid};
}

} // namespace linalg
} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/solve.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/vecmat.hpp
/**
 * @file
 * @brief Vector-matrix multiplication utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace linalg {
namespace detail {

template <md_c in1_t, md_c in2_t, md_c out_t>
inline constexpr void vecmat_naive_noalias(in1_t &&in1, in2_t &&in2,
                                           out_t &&out) noexcept {
    using out_index_t = typename std::remove_cvref_t<out_t>::index_type;
    using in1_index_t = typename std::remove_cvref_t<in1_t>::index_type;

    for (out_index_t i = 0; i < out.extent(0); i++) {
        out(i) = 0;

        for (in1_index_t j = 0; j < in1.extent(0); j++) {
            out(i) += in1(j) * in2(j, i);
        }
    }
}

template <md_c in1_t, md_c in2_t, md_c out_t>
inline constexpr void vecmat_naive(in1_t &&in1, in2_t &&in2,
                                   out_t &&out) noexcept {
    const auto in1_mds = core::to_const_mdspan(std::forward<in1_t>(in1));
    const auto in2_mds = core::to_const_mdspan(std::forward<in2_t>(in2));
    auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    if (std::is_constant_evaluated()) {
        auto out_tmp = empty_like(out_mds);
        vecmat_naive_noalias(in1_mds, in2_mds, out_tmp.to_mdspan());
        copy_to(out_tmp, out_mds);
        return;
    }

    bool need_copy = false;

    if constexpr (requires {
                      in1_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in1_mds.data_handle() == out_mds.data_handle()) [[unlikely]] {
            need_copy = true;
        }
    }

    if constexpr (requires {
                      in2_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in2_mds.data_handle() == out_mds.data_handle()) [[unlikely]] {
            need_copy = true;
        }
    }

    if (!need_copy) [[likely]] {
        vecmat_naive_noalias(in1_mds, in2_mds, out_mds);

    } else [[unlikely]] {
        auto out_tmp = empty_like(out_mds);
        vecmat_naive_noalias(in1_mds, in2_mds, out_tmp.to_mdspan());
        copy_to(out_tmp, out_mds);
    }
}

template <md_c in1_t, md_c in2_t, md_c out_t>
inline constexpr void vecmat_impl(in1_t &&in1, in2_t &&in2,
                                  out_t &&out) noexcept {
    static_assert(std::remove_cvref_t<in1_t>::rank() == 1);
    static_assert(std::remove_cvref_t<in2_t>::rank() == 2);
    static_assert(std::remove_cvref_t<out_t>::rank() == 1);

#ifdef MDTENSOR_USE_EIGEN
#if __cplusplus >= 202302L // TODO: Impliement for C++20
    if constexpr (core::eigen::eigen_mappable_mdspan_c<in1_t> &&
                  core::eigen::eigen_mappable_mdspan_c<in2_t> &&
                  core::eigen::eigen_mappable_mdspan_c<out_t>) {
        if (!std::is_constant_evaluated() && 8 <= out.extent(0) + out.extent(1))
            [[likely]] {
            using value_t = core::common_data_type_t<
                typename std::remove_cvref_t<in1_t>::value_type,
                typename std::remove_cvref_t<in2_t>::value_type>;

            const auto ein1 =
                core::eigen::to_eigen(in1).template cast<value_t>();
            const auto ein2 =
                core::eigen::to_eigen(in2).template cast<value_t>();
            auto eout = core::eigen::to_eigen(out);

            eout = (ein1 * ein2)
                       .template cast<
                           typename std::remove_cvref_t<out_t>::value_type>();

            return;
        }
    }

#else
    assert(false && "Eigen inverse not implemented for C++20");

#endif
#endif

    vecmat_naive(in1, in2, out);
}

} // namespace detail

/**
 * @brief Vector-matrix multiplication (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 Input vector (rank-1 mdspan, mdarray, etc.) (... x K).
 * @param in2 Input matrix (rank-2 mdspan, mdarray, etc.) (... x K x N).
 * @param out Output vector (rank-1 mdspan, mdarray, etc.) (... x N).
 *
 * @note The multiplication is performed per-instance when inputs are batched.
 *       The last dimension of `in1` must match the first dimension of `in2`.
 *
 * @see mdtensor::linalg::vecmat for the out-of-place version that returns the
 * result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void vecmat_to(in1_t &&in1, in2_t &&in2,
                                out_t &&out) noexcept {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::vecmat_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<1, 2, 1>{},
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Vector-matrix multiplication (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 * inputs.
 *
 * @param in1 Input vector (rank-1 mdspan, mdarray, etc.) (... x K).
 * @param in2 Input matrix (rank-2 mdspan, mdarray, etc.) (... x K x N).
 *
 * @return Vector (mdarray) with shape (... x N).
 *
 * @note The multiplication is performed per-instance when inputs are batched.
 *
 * @see mdtensor::linalg::vecmat_to for the in-place version that writes into an
 * output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto vecmat(in1_t &&in1, in2_t &&in2) noexcept {
    const auto in1_mds = core::to_const_mdspan(std::forward<in1_t>(in1));
    const auto in2_mds = core::to_const_mdspan(std::forward<in2_t>(in2));

    const auto uin1_exts = core::slice_from_right<2>(in1_mds.extents());
    const auto uin2_exts = core::slice_from_right<2>(in2_mds.extents());
    const auto uout_exts = extents<
        core::common_data_type_t<typename decltype(uin1_exts)::index_type,
                                 typename decltype(uin2_exts)::index_type>,
        decltype(uin2_exts)::static_extent(1)>{uin2_exts.extent(1)};

    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::vecmat_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<1, 2>{}, uout_exts, in1_mds, in2_mds);
}

} // namespace linalg

inline constexpr void vecmat_to(auto &&...elems) noexcept {
    linalg::vecmat_to(std::forward<decltype(elems)>(elems)...);
}

template <typename dtype = void>
[[nodiscard]] inline constexpr auto vecmat(auto &&...elems) noexcept {
    return linalg::vecmat<dtype>(std::forward<decltype(elems)>(elems)...);
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/vecmat.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/linalg.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/logic.hpp
/**
 * @file
 * @brief Comparison and logical utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/all.hpp
/**
 * @file
 * @brief Logical all-reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in_t, typename out_t>
inline constexpr void all_impl(in_t &&in, out_t &&out) {
    using in_base_t = std::remove_cvref_t<in_t>;
    using out_base_t = std::remove_cvref_t<out_t>;

    static_assert(in_base_t::rank() == out_base_t::rank() + 1,
                  "Input rank must be one greater than output rank.");

    if constexpr (in_base_t::rank() == 1) {
        for (typename in_base_t::index_type i = 0; i < in.extent(0); i++) {
            if (!static_cast<bool>(in(i))) {
                out() = false;
                return;
            }
        }

        out() = true;

    } else {
        for (typename in_base_t::index_type i = 0;
             i < in.extent(in_base_t::rank() - 1); i++) {
            all_impl(core::submdspan_from_right(std::forward<in_t>(in), i),
                     core::submdspan_from_right(std::forward<out_t>(out), i));
        }
    }
}

} // namespace detail

/**
 * @brief Compute logical all along a specified axis (in-place).
 *
 * @tparam Axis Axis to reduce. Negative values are supported and normalized
 *         by the input rank (NumPy-like semantics).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc. (rank reduced by 1).
 *
 * @note The output values are written as boolean results (true/false) into out.
 *
 * @see mdtensor::all<Axis, dtype>(in) for the out-of-place axis-reduction
 *      version.
 * @see mdtensor::all(in) for full-tensor reduction returning bool.
 */
template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename in_t,
          typename out_t>
inline constexpr void all(in_t &&in, out_t &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::all_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank, rin_rank - 1>{}, in_mds, out_mds);
}

/**
 * @brief Compute logical all along a specified axis (out-of-place).
 *
 * @tparam Axis Axis to reduce. Negative values are supported and normalized
 *         by the input rank (NumPy-like semantics).
 * @tparam dtype (optional) Data type of the result tensor. Default is int8_t.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return Reduced tensor (mdarray or scalar), with rank reduced by 1.
 *
 * @note dtype controls the storage type of the output tensor. The produced
 *       values represent boolean results (0/1) when dtype is integral.
 *
 * @see mdtensor::all<Axis>(in, out) for the in-place version.
 * @see mdtensor::all(in) for full-tensor reduction returning bool.
 */
template <int64_t Axis, typename dtype = int8_t, MPMode mpmode = MPMode::NONE,
          typename in_t>
[[nodiscard]] inline constexpr auto all(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::all_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank>{},
        core::slice_from_right<rin_rank - 1>(in_mds.extents()), in_mds);
}

/**
 * @brief Compute logical all over the entire input (full reduction).
 *
 * @tparam in_t Input type (mdspan, mdarray, scalar, etc.).
 *
 * @param in Input tensor-like object.
 *
 * @return true if all elements are truthy, otherwise false.
 *
 * @note For rank-0 inputs, this returns static_cast<bool>(in()).
 * @note For rank>0 inputs, the reduction is performed recursively.
 */
template <typename in_t> [[nodiscard]] inline constexpr bool all(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    if constexpr (decltype(in_mds)::rank() == 0) {
        return static_cast<bool>(in_mds());

    } else {
        using index_t = typename decltype(in_mds)::index_type;

        for (index_t i = 0; i < in_mds.extent(0); i++) {
            if (!all(core::submdspan_from_left(in_mds, i))) {
                return false;
            }
        }

        return true;
    }
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/all.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/allclose.hpp
/**
 * @file
 * @brief All-close comparison utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/isclose.hpp
/**
 * @file
 * @brief Element-wise approximate equality utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void isclose_impl(in1_t &&in1, in2_t &&in2, out_t &&out,
                                   const double &rtol = 1e-05,
                                   const double &atol = 1e-08) {
    out() = absolute(in1() - in2()) <=
            (atol + rtol * static_cast<double>(absolute(in2())));
}

} // namespace detail

/**
 * @brief Compare two inputs element-wise for approximate equality (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 * @param rtol Relative tolerance. Default is 1e-05.
 * @param atol Absolute tolerance. Default is 1e-08.
 *
 * @note Equivalent to out = (|in1 - in2| <= (atol + rtol * |in2|)) in terms of
 * array broadcasting.
 *
 * @see mdtensor::isclose for the out-of-place version that returns the result.
 * @see mdtensor::allclose for full-tensor reduction using isclose.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void isclose_to(in1_t &&in1, in2_t &&in2, out_t &&out,
                                 const double &rtol = 1e-05,
                                 const double &atol = 1e-08) {
    core::batch<mpmode>(
        [&](auto &&...elems) {
            detail::isclose_impl(std::forward<decltype(elems)>(elems)..., rtol,
                                 atol);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compare two inputs element-wise for approximate equality
 * (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. Default is int8_t.
 *         If an integral type is used, results represent boolean values (0/1).
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param rtol Relative tolerance. Default is 1e-05.
 * @param atol Absolute tolerance. Default is 1e-08.
 *
 * @return mdarray or scalar containing element-wise comparison results.
 *
 * @note Equivalent to out = (|in1 - in2| <= (atol + rtol * |in2|)) in terms of
 * array broadcasting.
 *
 * @see mdtensor::isclose_to for the in-place version that writes into an
 * output.
 * @see mdtensor::allclose for full-tensor reduction using isclose.
 */
template <typename dtype = int8_t, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto isclose(in1_t &&in1, in2_t &&in2,
                                            const double &rtol = 1e-05,
                                            const double &atol = 1e-08) {
    return core::batch_out<dtype, mpmode>(
        [&](auto &&...elems) {
            detail::isclose_impl(std::forward<decltype(elems)>(elems)..., rtol,
                                 atol);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/isclose.hpp

namespace mdtensor {

/**
 * @brief Return whether two inputs are element-wise close for all elements.
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam in1_t First input type (mdspan, mdarray, scalar, etc.).
 * @tparam in2_t Second input type (mdspan, mdarray, scalar, etc.).
 *
 * @param in1 First input tensor-like object.
 * @param in2 Second input tensor-like object.
 * @param rtol Relative tolerance.
 * @param atol Absolute tolerance.
 *
 * @return true if all elements satisfy isclose(in1, in2, rtol, atol),
 *         otherwise false.
 *
 * @note This is equivalent to all(isclose(in1, in2, rtol, atol)).
 * @note Broadcasting semantics follow those of mdtensor::isclose.
 *
 * @see mdtensor::isclose for the element-wise predicate.
 * @see mdtensor::all for logical reduction utilities.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t>
[[nodiscard]] inline constexpr bool allclose(in1_t &&in1, in2_t &&in2,
                                             const double &rtol = 1e-05,
                                             const double &atol = 1e-08) {
    return all(isclose<int8_t, mpmode>(std::forward<in1_t>(in1),
                                       std::forward<in2_t>(in2), rtol, atol));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/allclose.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/any.hpp
/**
 * @file
 * @brief Logical any-reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <md_c in_t, md_c out_t>
    requires(std::remove_cvref_t<out_t>::rank() == 0)
inline constexpr void any_impl(in_t &&in, out_t &&out) {
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    for (index_t i = 0; i < in.extent(0); i++) {
        if (static_cast<bool>(in(i))) {
            out() = true;
            return;
        }
    }

    out() = false;
}

template <md_c in_t, md_c out_t>
    requires(std::remove_cvref_t<out_t>::rank() > 0)
inline constexpr void any_impl(in_t &&in, out_t &&out) {
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    for (index_t i = 0; i < in.extent(in_base_t::rank() - 1); i++) {
        any_impl(core::submdspan_from_right(in, i),
                 core::submdspan_from_right(out, i));
    }
}

} // namespace detail

/**
 * @brief Compute logical any along a specified axis (in-place).
 *
 * @tparam Axis Axis to reduce. Negative values are supported and normalized
 *         by the input rank (NumPy-like semantics).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc. (rank reduced by 1).
 *
 * @note The output values are written as boolean results (true/false) into out.
 *
 * @see mdtensor::any<Axis, dtype>(in) for the out-of-place axis-reduction
 *      version.
 * @see mdtensor::any(in) for full-tensor reduction returning bool.
 */
template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename in_t,
          typename out_t>
inline constexpr void any(in_t &&in, out_t &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::any_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank, rin_rank - 1>{}, in_mds, out_mds);
}

/**
 * @brief Compute logical any along a specified axis (out-of-place).
 *
 * @tparam Axis Axis to reduce. Negative values are supported and normalized
 *         by the input rank (NumPy-like semantics).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result tensor. Default is int8_t.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return Reduced tensor (mdarray or scalar), with rank reduced by 1.
 *
 * @note dtype controls the storage type of the output tensor. The produced
 *       values represent boolean results (0/1) when dtype is integral.
 *
 * @see mdtensor::any<Axis>(in, out) for the in-place version.
 * @see mdtensor::any(in) for full-tensor reduction returning bool.
 */
template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename dtype = int8_t,
          typename in_t>
[[nodiscard]] inline constexpr auto any(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::any_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank>{},
        core::slice_from_right<rin_rank - 1>(in_mds.extents()), in_mds);
}

/**
 * @brief Compute logical any over the entire input (full reduction).
 *
 * @tparam in_t Input type (mdspan, mdarray, scalar, etc.).
 *
 * @param in Input tensor-like object.
 *
 * @return true if any element is truthy, otherwise false.
 *
 * @note For rank-0 inputs, this returns static_cast<bool>(in()).
 * @note For rank>0 inputs, the reduction is performed recursively.
 */
template <typename in_t> [[nodiscard]] inline constexpr bool any(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    if constexpr (in_mds_t::rank() == 0) {
        return static_cast<bool>(in_mds());

    } else {
        for (typename in_mds_t::index_type i = 0; i < in_mds.extent(0); i++) {
            if (any(core::submdspan_from_left(in_mds, i))) {
                return true;
            }
        }

        return false;
    }
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/any.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/array_equal.hpp
/**
 * @file
 * @brief Exact array equality utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

/**
 * @brief Return whether two inputs are exactly equal in shape and values.
 *
 * @tparam in1_t First input type (mdspan, mdarray, scalar, etc.).
 * @tparam in2_t Second input type (mdspan, mdarray, scalar, etc.).
 *
 * @param in1 First input tensor-like object.
 * @param in2 Second input tensor-like object.
 *
 * @return true if rank, extents, and all elements match exactly; otherwise
 *         false.
 *
 * @note This function does not apply broadcasting. Shapes must be identical.
 * @note For rank-0 inputs, this compares the two scalar values directly.
 * @note For rank>0 inputs, this checks extents and then recurses by slicing
 *       from the left.
 *
 * @see mdtensor::isclose for tolerant element-wise comparison.
 * @see mdtensor::allclose for tolerant full-tensor comparison.
 */
template <typename in1_t, typename in2_t>
[[nodiscard]] inline constexpr bool array_equal(in1_t &&in1, in2_t &&in2) {
    const auto in1_mds = core::to_const_mdspan(std::forward<in1_t>(in1));
    const auto in2_mds = core::to_const_mdspan(std::forward<in2_t>(in2));
    using in1_mds_t = decltype(in1_mds);
    using in2_mds_t = decltype(in2_mds);

    if constexpr (in1_mds_t::rank() != in2_mds_t::rank()) {
        return false;
    }

    for (size_t i = 0; i < in1_mds_t::rank(); i++) {
        if (in1_mds.extent(i) != in2_mds.extent(i)) {
            return false;
        }
    }

    if constexpr (in1_mds_t::rank() == 0) {
        return in1_mds() == in2_mds();

    } else {
        for (typename in1_mds_t::index_type i = 0; i < in1_mds.extent(0); i++) {
            if (!array_equal(core::submdspan_from_left(in1_mds, i),
                             core::submdspan_from_left(in2_mds, i))) {
                return false;
            }
        }
    }

    return true;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/array_equal.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/array_equiv.hpp
/**
 * @file
 * @brief Array equivalence utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/equal.hpp
/**
 * @file
 * @brief Element-wise equality utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void equal_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    out() = (in1() == in2());
}

} // namespace detail

/**
 * @brief Compare two inputs element-wise for equality (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = (in1 == in2) in terms of array broadcasting.
 *
 * @see mdtensor::equal for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void equal_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch(
        [](auto &&...elems) {
            detail::equal_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compare two inputs element-wise for equality (out-of-place).
 *
 * @tparam dtype (optional) Data type of the result. Default is int8_t.
 *         If an integral type is used, results represent boolean values (0/1).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar containing element-wise equality results.
 *
 * @note Equivalent to out = (in1 == in2) in terms of array broadcasting.
 *
 * @see mdtensor::equal_to for the in-place version that writes into an output.
 * @see mdtensor::array_equiv for full-tensor reduction with broadcasting.
 * @see mdtensor::array_equal for exact equality without broadcasting.
 */
template <typename dtype = int8_t, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto equal(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::equal_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/equal.hpp

namespace mdtensor {

/**
 * @brief Return whether two inputs are element-wise equal for all elements,
 *        with broadcasting.
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam in1_t First input type (mdspan, mdarray, scalar, etc.).
 * @tparam in2_t Second input type (mdspan, mdarray, scalar, etc.).
 *
 * @param in1 First input tensor-like object.
 * @param in2 Second input tensor-like object.
 *
 * @return true if all elements satisfy equal(in1, in2) after broadcasting;
 *         otherwise false.
 *
 * @note This is equivalent to all(equal(in1, in2)).
 * @note Broadcasting semantics follow those of mdtensor::equal.
 *
 * @see mdtensor::equal for the element-wise predicate.
 * @see mdtensor::all for logical reduction utilities.
 * @see mdtensor::array_equal for exact shape-and-value equality without
 *      broadcasting.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t>
[[nodiscard]] inline constexpr bool array_equiv(in1_t &&in1, in2_t &&in2) {
    return all(equal<int8_t, mpmode>(std::forward<in1_t>(in1),
                                     std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/array_equiv.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/greater.hpp
/**
 * @file
 * @brief Element-wise greater-than comparison utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void greater_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    out() = (in1() > in2());
}

} // namespace detail

/**
 * @brief Compare two inputs element-wise for greater-than (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = (in1 > in2) in terms of array broadcasting.
 *
 * @see mdtensor::greater for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void greater_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::greater_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compare two inputs element-wise for greater-than (out-of-place).
 *
 * @tparam dtype (optional) Data type of the result. Default is int8_t.
 *         If an integral type is used, results represent boolean values (0/1).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar containing element-wise comparison results.
 *
 * @note Equivalent to out = (in1 > in2) in terms of array broadcasting.
 *
 * @see mdtensor::greater_to for the in-place version that writes into an
 *      output.
 */
template <typename dtype = int8_t, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto greater(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::greater_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/greater.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/greater_equal.hpp
/**
 * @file
 * @brief Element-wise greater-or-equal comparison utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void greater_equal_impl(in1_t &&in1, in2_t &&in2,
                                         out_t &&out) {
    out() = (in1() >= in2());
}

} // namespace detail

/**
 * @brief Compare two inputs element-wise for greater-or-equal (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = (in1 >= in2) in terms of array broadcasting.
 *
 * @see mdtensor::greater_equal for the out-of-place version that returns the
 *      result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void greater_equal_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::greater_equal_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compare two inputs element-wise for greater-or-equal (out-of-place).
 *
 * @tparam dtype (optional) Data type of the result. Default is int8_t.
 *         If an integral type is used, results represent boolean values (0/1).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar containing element-wise comparison results.
 *
 * @note Equivalent to out = (in1 >= in2) in terms of array broadcasting.
 *
 * @see mdtensor::greater_equal_to for the in-place version that writes into an
 *      output.
 */
template <typename dtype = int8_t, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto greater_equal(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::greater_equal_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/greater_equal.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/less.hpp
/**
 * @file
 * @brief Element-wise less-than comparison utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void less_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    out() = (in1() < in2());
}

} // namespace detail

/**
 * @brief Compare two inputs element-wise for less-than (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = (in1 < in2) in terms of array broadcasting.
 *
 * @see mdtensor::less for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void less_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::less_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compare two inputs element-wise for less-than (out-of-place).
 *
 * @tparam dtype (optional) Data type of the result. Default is int8_t.
 *         If an integral type is used, results represent boolean values (0/1).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar containing element-wise comparison results.
 *
 * @note Equivalent to out = (in1 < in2) in terms of array broadcasting.
 *
 * @see mdtensor::less_to for the in-place version that writes into an output.
 */
template <typename dtype = int8_t, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto less(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::less_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/less.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/less_equal.hpp
/**
 * @file
 * @brief Element-wise less-or-equal comparison utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void less_equal_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    out() = (in1() <= in2());
}

} // namespace detail

/**
 * @brief Compare two inputs element-wise for less-or-equal (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = (in1 <= in2) in terms of array broadcasting.
 *
 * @see mdtensor::less_equal for the out-of-place version that returns the
 * result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void less_equal_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::less_equal_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compare two inputs element-wise for less-or-equal (out-of-place).
 *
 * @tparam dtype (optional) Data type of the result. Default is int8_t.
 *         If an integral type is used, results represent boolean values (0/1).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar containing element-wise comparison results.
 *
 * @note Equivalent to out = (in1 <= in2) in terms of array broadcasting.
 *
 * @see mdtensor::less_equal_to for the in-place version that writes into an
 * output.
 */
template <typename dtype = int8_t, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto less_equal(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::less_equal_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/less_equal.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/not_equal.hpp
/**
 * @file
 * @brief Element-wise not-equal comparison utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void not_equal_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    out() = (in1() != in2());
}

} // namespace detail

/**
 * @brief Compare two inputs element-wise for inequality (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = (in1 != in2) in terms of array broadcasting.
 *
 * @see mdtensor::not_equal for the out-of-place version that returns the
 * result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void not_equal_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::not_equal_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compare two inputs element-wise for inequality (out-of-place).
 *
 * @tparam dtype (optional) Data type of the result. Default is int8_t.
 *         If an integral type is used, results represent boolean values (0/1).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar containing element-wise comparison results.
 *
 * @note Equivalent to out = (in1 != in2) in terms of array broadcasting.
 *
 * @see mdtensor::not_equal_to for the in-place version that writes into an
 * output.
 */
template <typename dtype = int8_t, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto not_equal(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::not_equal_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/not_equal.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/logic.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/manipulation.hpp
/**
 * @file
 * @brief Manipulation utilities header aggregator for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/concatenate.hpp
/**
 * @file
 * @brief Concatenation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/expand_dims.hpp
/**
 * @file
 * @brief Dimension expansion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/reshape.hpp
/**
 * @file
 * @brief Reshape utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

/**
 * @brief Reshape a tensor into new extents (view-only).
 *
 * @tparam exts_t Extents type satisfying extents_c.
 * @tparam in_t Input type (mdspan, mdarray, scalar, etc.).
 *
 * @param in Input tensor-like object.
 * @param new_exts Target extents (shape). Default constructs exts_t if omitted.
 *
 * @return A reshaped mdspan view sharing the same underlying storage.
 *
 * @note This function does not allocate new memory.
 * @note The total number of elements must match between the input and the
 *       requested extents, as enforced by core::reshape.
 *
 * @see mdtensor::expand_dims for inserting singleton dimensions.
 * @see mdtensor::concatenate for joining tensors along an axis.
 */
template <extents_c exts_t, typename in_t>
[[nodiscard]] inline constexpr auto
reshape(in_t &&in, exts_t &&new_exts = exts_t{}) noexcept {
    return core::reshape(core::to_mdspan(std::forward<in_t>(in)),
                         std::forward<exts_t>(new_exts));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/reshape.hpp

namespace mdtensor {

/**
 * @brief Insert a new axis of length 1 into the input (out-of-place view).
 *
 * @tparam Axis Axis position to insert the new dimension. Negative values are
 *         supported and normalized (NumPy-like semantics).
 * @tparam in_t Input type (mdspan, mdarray, scalar, etc.).
 *
 * @param in Input tensor-like object.
 *
 * @return A reshaped view with rank increased by 1.
 *
 * @note This function does not allocate new storage; it returns a view
 *       (via reshape or direct mdspan construction for rank-0).
 * @note For rank-0 inputs, the resulting shape is (1).
 * @note For rank>0 inputs, the new shape matches the input with a singleton
 *       inserted at Axis.
 *
 * @see mdtensor::reshape for general reshaping utilities.
 */
template <int64_t Axis, typename in_t>
[[nodiscard]] inline constexpr auto expand_dims(in_t &&in) noexcept {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    constexpr size_t rank = in_mds_t::rank();

    if constexpr (rank == 0) {
        auto new_extents = extents<typename in_mds_t::index_type, 1>{1};
        return mdspan<typename in_mds_t::element_type, decltype(new_extents)>{
            in_mds.data_handle(), new_extents};

    } else {
        constexpr size_t axis = static_cast<size_t>(
            ((Axis % static_cast<int64_t>(rank + 1)) + (rank + 1)) %
            (rank + 1));

        const auto new_extents = [&in_mds]<size_t... Is>(
                                     std::index_sequence<Is...>) {
            return extents<
                typename in_mds_t::index_type,
                (Is < axis
                     ? in_mds_t::static_extent(Is)
                     : (Is == axis ? 1 : in_mds_t::static_extent(Is - 1)))...>{
                (Is < axis ? in_mds.extent(Is)
                           : (Is == axis ? 1 : in_mds.extent(Is - 1)))...};
        }(std::make_index_sequence<rank + 1>{});

        return reshape(std::forward<in_t>(in), new_extents);
    }
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/expand_dims.hpp

namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, size_t Is, size_t axis>
constexpr size_t concatenate_static_extent() {
    constexpr size_t e1 = in1_t::static_extent(Is);
    constexpr size_t e2 = in2_t::static_extent(Is);

    if constexpr (Is == axis) {
        return (e1 != dyn && e2 != dyn) ? (e1 + e2) : dyn;

    } else {
        static_assert(e1 == e2 || e1 == dyn || e2 == dyn,
                      "Incompatible extents for concatenate.");
        return (e1 != dyn && e2 != dyn) ? e1 : dyn;
    }
}

template <int64_t Axis, extents_c in1_t, extents_c in2_t, extents_c... ins_t>
    requires(in1_t::rank() == in2_t::rank())
[[nodiscard]] inline constexpr auto
concatenate_extents(const in1_t &in1 = in1_t{}, const in2_t &in2 = in2_t{},
                    const ins_t &...ins) noexcept {
    constexpr size_t rank = in1_t::rank();
    constexpr size_t axis = static_cast<size_t>(
        ((Axis % static_cast<int64_t>(rank)) + (rank)) % rank);

    using index_t = core::common_index_type_t<typename in1_t::index_type,
                                              typename in2_t::index_type>;

    const auto exts = [&]<typename E1, typename E2>(const E1 &e1,
                                                    const E2 &e2) {
        static_assert(E1::rank() == E2::rank());

        auto dyn_extent = [&]<std::size_t I>() -> index_t {
            const index_t d1 = static_cast<index_t>(e1.extent(I));
            const index_t d2 = static_cast<index_t>(e2.extent(I));

            if constexpr (I == axis) {
                return static_cast<index_t>(d1 + d2);

            } else {
                assert(d1 == d2);
                return d1;
            }
        };

        auto impl = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return extents<index_t,
                           concatenate_static_extent<E1, E2, Is, axis>()...>{
                dyn_extent.template operator()<Is>()...};
        };

        return impl(std::make_index_sequence<E1::rank()>{});
    }(in1, in2);

    if constexpr (sizeof...(ins_t) != 0) {
        return concatenate_extents<axis>(exts, ins...);

    } else {
        return exts;
    }
}

template <int64_t Axis, extents_c... ins_t>
[[nodiscard]] inline constexpr auto
concatenate_extents(std::tuple<ins_t...> &&ins) noexcept {
    constexpr size_t ins_num =
        std::tuple_size_v<std::remove_reference_t<decltype(ins)>>;

    static_assert(ins_num != 0, "concatenate requires at least one input.");

    if constexpr (ins_num == 1) {
        return std::get<0>(ins);

    } else {
        return std::apply(
            [&](auto &&...elems) {
                return concatenate_extents<Axis>(
                    std::forward<decltype(elems)>(elems)...);
            },
            ins);
    }
}

} // namespace detail

/**
 * @brief Concatenate multiple inputs along a specified axis (out-of-place).
 *
 * @tparam Axis Axis to concatenate. Negative values are supported and
 *         normalized by the input rank (NumPy-like semantics).
 * @tparam ins_t Input types (mdspan, mdarray, scalar, etc.).
 *
 * @param ins Input tensors to concatenate.
 *
 * @return Newly allocated mdarray containing the concatenation result.
 *
 * @note Rank-0 inputs (scalars) are expanded to rank-1 along the last axis
 *       using expand_dims<-1>.
 * @note All inputs must have the same rank after scalar expansion.
 * @note For dimensions other than Axis, extents must match.
 * @note The output dtype is the common_type of all input value types.
 */
template <int64_t Axis, typename... ins_t>
[[nodiscard]] inline constexpr auto concatenate(ins_t &&...ins) noexcept {
    constexpr size_t num_ins = sizeof...(ins_t);
    auto ins_tuple = std::forward_as_tuple(ins...);

    // generate input mdspans
    const auto ins_mds = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::make_tuple([&]() {
            auto mds = core::to_const_mdspan(std::get<Is>(ins_tuple));

            if constexpr (mds.rank() == 0) {
                // if the input is a scalar, expand it to a 1D mdspan.
                return expand_dims<-1>(mds);

            } else {
                return mds;
            }
        }()...);
    }(std::make_index_sequence<num_ins>{});

    using ins_mds_t = decltype(ins_mds);

    // generate out extents
    constexpr size_t rank = std::tuple_element_t<0, ins_mds_t>::rank();
    constexpr size_t axis = static_cast<size_t>(
        ((Axis % static_cast<int64_t>(rank)) + (rank)) % rank);

    const auto out_extents =
        [&ins_mds]<size_t... Is>(std::index_sequence<Is...>) {
            return detail::concatenate_extents<axis>(
                std::make_tuple(std::get<Is>(ins_mds).extents()...));
        }(std::make_index_sequence<num_ins>{});

    // generate out
    using dtype = decltype([]<size_t... Is>(std::index_sequence<Is...>) {
        return core::common_data_type_t<
            core::value_type_t<std::tuple_element_t<Is, ins_mds_t>>...>{};
    }(std::make_index_sequence<num_ins>{}));
    auto out = empty<dtype>(out_extents);

    // concatenate
    [&ins_mds, &out]<size_t... Is>(std::index_sequence<Is...>) {
        (([&] {
             const size_t offset =
                 [&]<size_t... Js>(std::index_sequence<Js...>) {
                     return (0 + ... + std::get<Js>(ins_mds).extent(axis));
                 }(std::make_index_sequence<Is>{});
             const size_t extent = std::get<Is>(ins_mds).extent(axis);
             constexpr size_t stride = 1;

             copy_to(std::get<Is>(ins_mds),
                     [&]<size_t... Js>(std::index_sequence<Js...>) {
                         return core::submdspan_from_left(
                             core::to_mdspan(out), ((void)Js, full_extent)...,
                             strided_slice{offset, extent, stride});
                     }(std::make_index_sequence<axis>{}));
         })(),
         ...);
    }(std::make_index_sequence<num_ins>{});

    return out;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/concatenate.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/transpose.hpp
/**
 * @file
 * @brief Transpose utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


#include <array>
#include <type_traits>
#include <utility>


namespace mdtensor {

template <typename in_t, std::integral AxesType, AxesType... Axes>
[[nodiscard]] inline constexpr auto
transpose(in_t &&in, std::integer_sequence<AxesType, Axes...>) noexcept {
    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    constexpr size_t rank = in_mds_t::rank();

    static_assert(sizeof...(Axes) == rank, "Number of axes must match rank.");

    if constexpr (rank == 0 || rank == 1) {
        return in_mds;

    } else {
        constexpr auto axes = std::array<size_t, rank>{static_cast<size_t>(
            ((Axes % static_cast<AxesType>(rank)) + (rank)) % rank)...};

        const auto new_extents = [&]<size_t... Is>(std::index_sequence<Is...>) {
            return extents<typename in_mds_t::index_type,
                           in_mds_t::static_extent(axes[Is])...>{
                in_mds.extent(axes[Is])...};
        }(std::make_index_sequence<rank>{});

        const auto new_strides = [&]<size_t... Is>(std::index_sequence<Is...>) {
            return std::array<typename in_mds_t::index_type, rank>{
                in_mds.stride(axes[Is])...};
        }(std::make_index_sequence<rank>{});

        return mdspan<typename in_mds_t::element_type,
                      std::remove_cvref_t<decltype(new_extents)>, layout_stride,
                      typename in_mds_t::accessor_type>{
            in_mds.data_handle(),
            layout_stride::mapping{new_extents, new_strides}};
    }
}

template <typename in_t>
[[nodiscard]] inline constexpr auto transpose(in_t &&in) noexcept {
    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    constexpr size_t rank = in_mds_t::rank();

    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return transpose(std::forward<in_t>(in),
                         std::integer_sequence<int64_t, rank - 1 - Is...>{});
    }(std::make_index_sequence<rank>{});
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/transpose.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/manipulation.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/math.hpp
/**
 * @file
 * @brief Math utilities header aggregator for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/atan2.hpp
/**
 * @file
 * @brief Element-wise atan2 utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void atan2_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    using value_t = core::common_data_type_t<decltype(in1()), decltype(in2()),
                                             decltype(out())>;

    out() =
        std::atan2(static_cast<value_t>(in1()), static_cast<value_t>(in2()));
}

} // namespace detail

/**
 * @brief Compute atan2 element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc. (y-coordinate).
 * @param in2 Second input mdspan, mdarray, scalar, etc. (x-coordinate).
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = std::atan2(in1, in2) in terms of array
 *       broadcasting.
 *
 * @see mdtensor::atan2 for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void atan2_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::atan2_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compute atan2 element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         inputs.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc. (y-coordinate).
 * @param in2 Second input mdspan, mdarray, scalar, etc. (x-coordinate).
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = std::atan2(in1, in2) in terms of array
 *       broadcasting.
 *
 * @see mdtensor::atan2_to for the in-place version that writes into an output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto atan2(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::atan2_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/atan2.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/clip.hpp
/**
 * @file
 * @brief Element-wise clipping utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in_t, typename min_t, typename max_t, typename out_t>
inline constexpr void clip_impl(in_t &&in, min_t &&min, max_t &&max,
                                out_t &&out) {
    // NOTE: std::clamp is not used to match the behavior with original np.clip
    out() = in();

    if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(min())>,
                                  std::nullopt_t>) {
        using value_t =
            core::common_data_type_t<std::remove_cvref_t<decltype(out())>,
                                     std::remove_cvref_t<decltype(min())>>;

        out() =
            std::max(static_cast<value_t>(out()), static_cast<value_t>(min()));
    }

    if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(max())>,
                                  std::nullopt_t>) {
        using value_t =
            core::common_data_type_t<std::remove_cvref_t<decltype(out())>,
                                     std::remove_cvref_t<decltype(max())>>;

        out() =
            std::min(static_cast<value_t>(out()), static_cast<value_t>(max()));
    }
}

} // namespace detail

/**
 * @brief Clip input values to the given range element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param min Minimum bound (mdspan, mdarray, scalar, etc.) or a disabled bound.
 * @param max Maximum bound (mdspan, mdarray, scalar, etc.) or a disabled bound.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = clip(in, min, max) in terms of array broadcasting.
 * @note Either bound may be disabled by passing a "no-bound" value (e.g.,
 *       nullopt wrapper), in which case only the other bound is applied.
 *
 * @see mdtensor::clip for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename min_t,
          typename max_t, typename out_t>
inline constexpr void clip_to(in_t &&in, min_t &&min, max_t &&max,
                              out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::clip_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_const_mdspan(std::forward<min_t>(min)),
        core::to_const_mdspan(std::forward<max_t>(max)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Clip input values to the given range element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         inputs.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param min Minimum bound (mdspan, mdarray, scalar, etc.) or a disabled bound.
 * @param max Maximum bound (mdspan, mdarray, scalar, etc.) or a disabled bound.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = clip(in, min, max) in terms of array broadcasting.
 * @note Either bound may be disabled by passing a "no-bound" value (e.g.,
 *       nullopt wrapper), in which case only the other bound is applied.
 *
 * @see mdtensor::clip_to for the in-place version that writes into an output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t,
          typename min_t, typename max_t>
[[nodiscard]] inline constexpr auto clip(in_t &&in, min_t &&min, max_t &&max) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::clip_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_const_mdspan(std::forward<min_t>(min)),
        core::to_const_mdspan(std::forward<max_t>(max)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/clip.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/cos.hpp
/**
 * @file
 * @brief Element-wise cosine utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in_t, typename out_t>
inline constexpr void cos_impl(in_t &&in, out_t &&out) {
    out() = std::cos(in());
}

} // namespace detail

/**
 * @brief Compute cosine element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = std::cos(in) in terms of array broadcasting.
 *
 * @see mdtensor::cos for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void cos_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::cos_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compute cosine element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = std::cos(in) in terms of array broadcasting.
 *
 * @see mdtensor::cos_to for the in-place version that writes into an output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto cos(in_t &&in) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::cos_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/cos.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/deg2rad.hpp
/**
 * @file
 * @brief Degree-to-radian conversion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


#include <numbers>


namespace mdtensor {

/**
 * @brief Convert degrees to radians element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input angles in degrees (mdspan, mdarray, scalar, etc.).
 * @param out Output angles in radians (mdspan, mdarray, scalar, etc.).
 *
 * @note Equivalent to out = in * (pi / 180) in terms of array broadcasting.
 * @note The scaling constant is computed in a common type of the input value
 *       type and float to ensure floating-point conversion.
 *
 * @see mdtensor::deg2rad for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void deg2rad_to(in_t &&in, out_t &&out) {
    using value_t =
        core::common_data_type_t<typename decltype(core::to_mdspan(
                                     std::forward<in_t>(in)))::value_type,
                                 float>;

    constexpr value_t D2R = std::numbers::pi_v<value_t> / value_t(180);

    multiply_to<mpmode>(std::forward<in_t>(in), D2R, std::forward<out_t>(out));
}

/**
 * @brief Convert degrees to radians element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input angles in degrees (mdspan, mdarray, scalar, etc.).
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = in * (pi / 180) in terms of array broadcasting.
 * @note The scaling constant is computed in a common type of the input value
 *       type and float to ensure floating-point conversion.
 *
 * @see mdtensor::deg2rad_to for the in-place version that writes into an
 *      output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto deg2rad(in_t &&in) {
    using value_t =
        core::common_data_type_t<typename decltype(core::to_mdspan(
                                     std::forward<in_t>(in)))::value_type,
                                 float>;

    constexpr value_t D2R = std::numbers::pi_v<value_t> / value_t(180);

    return multiply<dtype, mpmode>(std::forward<in_t>(in), D2R);
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/deg2rad.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/divide.hpp
/**
 * @file
 * @brief Element-wise division utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void divide_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    out() = in1() / in2();
}

} // namespace detail

/**
 * @brief Divide arguments element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = in1 / in2 in terms of array broadcasting.
 *
 * @see mdtensor::divide for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void divide_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::divide_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Divide arguments element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         inputs.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = in1 / in2 in terms of array broadcasting.
 *
 * @see mdtensor::divide_to for the in-place version that writes into an output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto divide(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::divide_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/divide.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/max.hpp
/**
 * @file
 * @brief Maximum reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <md_c in_t, md_c out_t>
    requires(std::remove_cvref_t<out_t>::rank() == 0)
inline constexpr void max_impl(in_t &&in, out_t &&out) {
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    out() = in(0);

    for (index_t i = 1; i < in.extent(0); i++) {
        if (in(i) > out()) {
            out() = in(i);
        }
    }
}

template <md_c in_t, md_c out_t>
    requires(std::remove_cvref_t<out_t>::rank() > 0)
inline constexpr void max_impl(in_t &&in, out_t &&out) {
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    for (index_t i = 0; i < in.extent(in_base_t::rank() - 1); i++) {
        max_impl(core::submdspan_from_right(in, i),
                 core::submdspan_from_right(out, i));
    }
}

} // namespace detail

namespace {

template <md_c in_t, arithmetic_c out_t>
inline constexpr void max_arithmetic_impl(const in_t &in, out_t &out) {
    // NOTE: CANNOT USE THIS FUNCTION DIRECTLY. OUT SHOULD BE INITIALIZED
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    if constexpr (in_base_t::rank() == 0) {
        if (in() > out) {
            out = in();
        }

    } else {
        for (index_t i = 0; i < in.extent(0); i++) {
            max_arithmetic_impl(core::submdspan_from_left(in, i), out);
        }
    }
}

} // namespace

/**
 * @brief Compute maximum along a specified axis (in-place).
 *
 * @tparam Axis Axis to reduce. Negative values are supported and normalized
 *         by the input rank (NumPy-like semantics).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note The reduction is performed along the specified axis and written to out.
 * @note Broadcasting is not performed; this is a reduction operation.
 *
 * @see mdtensor::max for the out-of-place axis-reduction version that returns
 *      the result.
 */
template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename in_t,
          typename out_t>
inline constexpr void max_to(in_t &&in, out_t &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::max_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank, rin_rank - 1>{}, in_mds, out_mds);
}

/**
 * @brief Compute maximum along a specified axis (out-of-place).
 *
 * @tparam Axis Axis to reduce. Negative values are supported and normalized
 *         by the input rank (NumPy-like semantics).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar (depending on the input rank and reduction).
 *
 * @note The reduction is performed along the specified axis.
 *
 * @see mdtensor::max_to for the in-place version that writes into an output.
 */
template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename dtype = void,
          typename in_t>
[[nodiscard]] inline constexpr auto max(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::max_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank>{},
        core::slice_from_right<rin_rank - 1>(in_mds.extents()), in_mds);
}

/**
 * @brief Compute maximum over all elements (full reduction).
 *
 * @tparam in_t Input type.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return Maximum value as an arithmetic scalar.
 *
 * @note This overload reduces all elements to a single scalar value.
 * @note The accumulator is initialized to the lowest representable value.
 */
template <typename in_t> [[nodiscard]] inline constexpr auto max(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    typename in_mds_t::value_type out =
        std::numeric_limits<typename in_mds_t::value_type>::lowest();

    max_arithmetic_impl(in_mds, out);

    return out;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/max.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/maximum.hpp
/**
 * @file
 * @brief Element-wise maximum utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void maximum_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    using value_t = std::remove_cvref_t<decltype(out())>;

    out() = std::max(static_cast<value_t>(in1()), static_cast<value_t>(in2()));
}

} // namespace detail

/**
 * @brief Compute element-wise maximum of two inputs (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = std::max(in1, in2) in terms of array broadcasting.
 *
 * @see mdtensor::maximum for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void maximum_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::maximum_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compute element-wise maximum of two inputs (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         inputs.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = std::max(in1, in2) in terms of array broadcasting.
 *
 * @see mdtensor::maximum_to for the in-place version that writes into an
 *      output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto maximum(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::maximum_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/maximum.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/min.hpp
/**
 * @file
 * @brief Minimum reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <md_c in_t, md_c out_t>
    requires(std::remove_cvref_t<out_t>::rank() == 0)
inline constexpr void min_impl(in_t &&in, out_t &&out) {
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    out() = in(0);

    for (index_t i = 1; i < in.extent(0); i++) {
        if (in(i) < out()) {
            out() = in(i);
        }
    }
}

template <md_c in_t, md_c out_t>
    requires(std::remove_cvref_t<out_t>::rank() > 0)
inline constexpr void min_impl(in_t &&in, out_t &&out) {
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    for (index_t i = 0; i < in.extent(in_base_t::rank() - 1); i++) {
        min_impl(core::submdspan_from_right(in, i),
                 core::submdspan_from_right(out, i));
    }
}

} // namespace detail

namespace {

template <md_c in_t, arithmetic_c out_t>
inline constexpr void min_arithmetic_impl(const in_t &in, out_t &out) {
    // NOTE: CANNOT USE THIS FUNCTION DIRECTLY. OUT SHOULD BE INITIALIZED
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    if constexpr (in_base_t::rank() == 0) {
        if (in() < out) {
            out = in();
        }

    } else {
        for (index_t i = 0; i < in.extent(0); i++) {
            min_arithmetic_impl(core::submdspan_from_left(in, i), out);
        }
    }
}

} // namespace

/**
 * @brief Compute minimum along a specified axis (in-place).
 *
 * @tparam Axis Axis to reduce. Negative values are supported and normalized
 *         by the input rank (NumPy-like semantics).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note The reduction is performed along the specified axis and written to out.
 * @note Broadcasting is not performed; this is a reduction operation.
 *
 * @see mdtensor::min for the out-of-place axis-reduction version that returns
 *      the result.
 */
template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename in_t,
          typename out_t>
inline constexpr void min_to(in_t &&in, out_t &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::min_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank, rin_rank - 1>{}, in_mds, out_mds);
}

/**
 * @brief Compute minimum along a specified axis (out-of-place).
 *
 * @tparam Axis Axis to reduce. Negative values are supported and normalized
 *         by the input rank (NumPy-like semantics).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar (depending on the input rank and reduction).
 *
 * @note The reduction is performed along the specified axis.
 *
 * @see mdtensor::min_to for the in-place version that writes into an output.
 */
template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename dtype = void,
          typename in_t>
[[nodiscard]] inline constexpr auto min(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::min_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank>{},
        core::slice_from_right<rin_rank - 1>(in_mds.extents()), in_mds);
}

/**
 * @brief Compute minimum over all elements (full reduction).
 *
 * @tparam in_t Input type.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return Minimum value as an arithmetic scalar.
 *
 * @note This overload reduces all elements to a single scalar value.
 * @note The accumulator is initialized to the highest representable value.
 */
template <typename in_t> [[nodiscard]] inline constexpr auto min(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    typename in_mds_t::value_type out =
        std::numeric_limits<typename in_mds_t::value_type>::max();

    min_arithmetic_impl(in_mds, out);

    return out;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/min.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/minimum.hpp
/**
 * @file
 * @brief Element-wise minimum utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void minimum_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    using value_t = std::remove_cvref_t<decltype(out())>;

    out() = std::min(static_cast<value_t>(in1()), static_cast<value_t>(in2()));
}

} // namespace detail

/**
 * @brief Compute element-wise minimum of two inputs (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = std::min(in1, in2) in terms of array broadcasting.
 *
 * @see mdtensor::minimum for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void minimum_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::minimum_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compute element-wise minimum of two inputs (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         inputs.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = std::min(in1, in2) in terms of array broadcasting.
 *
 * @see mdtensor::minimum_to for the in-place version that writes into an
 *      output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto minimum(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::minimum_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/minimum.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/negative.hpp
/**
 * @file
 * @brief Element-wise negation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in_t, typename out_t>
inline constexpr void negative_impl(in_t &&in, out_t &&out) {
    out() = -in();
}

} // namespace detail

/**
 * @brief Compute unary negation element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = -in in terms of array broadcasting.
 *
 * @see mdtensor::negative for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void negative_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::negative_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compute unary negation element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = -in in terms of array broadcasting.
 *
 * @see mdtensor::negative_to for the in-place version that writes into an
 *      output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto negative(in_t &&in) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::negative_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/negative.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/rad2deg.hpp
/**
 * @file
 * @brief Radian-to-degree conversion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


#include <numbers>


namespace mdtensor {

/**
 * @brief Convert radians to degrees element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input angles in radians (mdspan, mdarray, scalar, etc.).
 * @param out Output angles in degrees (mdspan, mdarray, scalar, etc.).
 *
 * @note Equivalent to out = in * (180 / pi) in terms of array broadcasting.
 * @note The scaling constant is computed in a common type of the input value
 *       type and float to ensure floating-point conversion.
 *
 * @see mdtensor::rad2deg for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void rad2deg_to(in_t &&in, out_t &&out) {
    using value_t =
        core::common_data_type_t<typename decltype(core::to_mdspan(
                                     std::forward<in_t>(in)))::value_type,
                                 float>;

    constexpr value_t R2D = std::numbers::inv_pi_v<value_t> * value_t(180);

    multiply_to<mpmode>(std::forward<in_t>(in), R2D, std::forward<out_t>(out));
}

/**
 * @brief Convert radians to degrees element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input angles in radians (mdspan, mdarray, scalar, etc.).
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = in * (180 / pi) in terms of array broadcasting.
 * @note The scaling constant is computed in a common type of the input value
 *       type and float to ensure floating-point conversion.
 *
 * @see mdtensor::rad2deg_to for the in-place version that writes into an
 *      output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto rad2deg(in_t &&in) {
    using value_t =
        core::common_data_type_t<typename decltype(core::to_mdspan(
                                     std::forward<in_t>(in)))::value_type,
                                 float>;

    constexpr value_t R2D = std::numbers::inv_pi_v<value_t> * value_t(180);

    return multiply<dtype, mpmode>(std::forward<in_t>(in), R2D);
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/rad2deg.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/sign.hpp
/**
 * @file
 * @brief Element-wise sign utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in_t, typename out_t>
inline constexpr void sign_impl(in_t &&in, out_t &&out) {
    out() = (in() > 0) - (in() < 0);
}

} // namespace detail

/**
 * @brief Compute sign element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @see mdtensor::sign for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void sign_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::sign_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compute sign element-wise (out-of-place).
 *
 * @tparam dtype (optional) Data type of the result. Default is int8_t.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note By default, the result type is int8_t to represent {-1, 0, 1}.
 *
 * @see mdtensor::sign_to for the in-place version that writes into an output.
 */
template <typename dtype = int8_t, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto sign(in_t &&in) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::sign_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/sign.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/sin.hpp
/**
 * @file
 * @brief Element-wise sine utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in_t, typename out_t>
inline constexpr void sin_impl(in_t &&in, out_t &&out) {
    out() = std::sin(in());
}

} // namespace detail

/**
 * @brief Compute sine element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = std::sin(in) in terms of array broadcasting.
 *
 * @see mdtensor::sin for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void sin_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::sin_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compute sine element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = std::sin(in) in terms of array broadcasting.
 *
 * @see mdtensor::sin_to for the in-place version that writes into an output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto sin(in_t &&in) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::sin_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/sin.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/subtract.hpp
/**
 * @file
 * @brief Element-wise subtraction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void subtract_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    out() = in1() - in2();
}

} // namespace detail

/**
 * @brief Subtract arguments element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = in1 - in2 in terms of array broadcasting.
 *
 * @see mdtensor::subtract for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void subtract_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::subtract_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Subtract arguments element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         inputs.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = in1 - in2 in terms of array broadcasting.
 *
 * @see mdtensor::subtract_to for the in-place version that writes into an
 *      output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto subtract(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::subtract_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/subtract.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/tan.hpp
/**
 * @file
 * @brief Element-wise tangent utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in_t, typename out_t>
inline constexpr void tan_impl(in_t &&in, out_t &&out) {
    out() = std::tan(in());
}

} // namespace detail

/**
 * @brief Compute tangent element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = std::tan(in) in terms of array broadcasting.
 *
 * @see mdtensor::tan for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void tan_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::tan_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compute tangent element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = std::tan(in) in terms of array broadcasting.
 *
 * @see mdtensor::tan_to for the in-place version that writes into an output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto tan(in_t &&in) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::tan_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/tan.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/math.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/random.hpp
/**
 * @file
 * @brief Random distribution header aggregator for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/rand.hpp
/**
 * @file
 * @brief Random number generation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


#include <cstdint>
#include <random>


namespace mdtensor {
namespace random {
namespace detail {

// Code adapted from:
// https://mklimenko.github.io/english/2018/06/04/constexpr-random/

constexpr std::uint32_t lce_a = 16807;
constexpr std::uint32_t lce_c = 0;
constexpr std::uint32_t lce_m = 2147483647;

[[nodiscard]] inline constexpr std::uint32_t time_from_string(const char *str,
                                                              int offset) {
    return static_cast<std::uint32_t>(str[offset] - '0') * 10 +
           static_cast<std::uint32_t>(str[offset + 1] - '0');
}

[[nodiscard]] inline constexpr std::uint32_t get_seed_constexpr() {
    const char *t = __TIME__;
    return time_from_string(t, 0) * 3600 + time_from_string(t, 3) * 60 +
           time_from_string(t, 6);
}

struct LCEngine {
    std::uint32_t state;

    constexpr LCEngine(std::uint32_t seed) : state(seed) {}

    [[nodiscard]] constexpr std::uint32_t next() {
        state = (lce_a * state + lce_c) % lce_m;
        return state;
    }

    [[nodiscard]] constexpr double next_normalized() {
        return static_cast<double>(next()) / lce_m;
    }
};

template <typename T, std::size_t sz>
[[nodiscard]] inline constexpr auto uniform_distribution(T min, T max) {
    std::array<T, sz> dst{};
    LCEngine rng{get_seed_constexpr()};
    for (auto &el : dst)
        el = static_cast<T>(rng.next_normalized() * (max - min) + min);
    return dst;
}

template <md_c in_t>
    requires(
        std::remove_cvref_t<in_t>::rank() == 0 &&
        std::floating_point<typename std::remove_cvref_t<in_t>::value_type>)
inline void rand_impl(in_t &&in) noexcept {
    using dist_t = std::uniform_real_distribution<
        typename std::remove_cvref_t<in_t>::value_type>;

    static auto rd = std::random_device{};
    thread_local static auto gen = std::mt19937(rd());
    static auto dist = dist_t{0, 1};

    in() = dist(gen);
}

} // namespace detail

/**
 * @brief Fill an output tensor with uniform random values in [0, 1).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Output mdspan, mdarray, scalar, etc.
 *
 * @note For constant evaluation with fully static extents, a constexpr
 *       LCEngine-based generator is used and the data is copied into the
 *       output.
 * @note For runtime execution, the core batch engine fills elements using a
 *       thread-local std::mt19937 distribution on [0, 1).
 *
 * @see mdtensor::random::rand for the out-of-place version that allocates an
 *      output and fills it.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t>
inline constexpr void rand_to(in_t &&in) noexcept {
    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    if constexpr (in_mds_t::rank_dynamic() == 0) {
        if (std::is_constant_evaluated()) {
            using T = core::value_type_t<in_mds_t>;

            if constexpr (in_mds_t::rank() == 0) {
                constexpr auto data =
                    detail::uniform_distribution<T, 1>(0, 1)[0];
                in_mds() = data;

            } else {
                constexpr auto data_size =
                    []<size_t... Is>(std::index_sequence<Is...>) {
                        return (in_mds_t::static_extent(Is) * ...);
                    }(std::make_index_sequence<in_mds_t::rank()>{});

                constexpr auto data =
                    mdarray<T, typename in_mds_t::extents_type>{
                        detail::uniform_distribution<T, data_size>(0, 1)};

                copy_to(data, in_mds);
            }

            return;
        }
    }

    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::rand_impl(std::forward<decltype(elems)>(elems)...);
        },
        in_mds);
}

/**
 * @brief Create an array of uniform random values in [0, 1).
 *
 * @tparam dtype (optional) Floating-point element type. Default is float.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam exts_t (optional) Extents type. Default is extents<uint8_t>.
 *
 * @param exts Output extents.
 *
 * @return Newly allocated mdarray filled with uniform random values in [0, 1).
 *
 * @see mdtensor::random::rand_to for the in-place version that fills an
 *      existing output.
 */
template <std::floating_point dtype = float, MPMode mpmode = MPMode::NONE,
          extents_c exts_t = extents<uint8_t>>
[[nodiscard]] inline constexpr auto rand(exts_t &&exts = exts_t{}) noexcept {
    auto out = empty<dtype>(std::forward<exts_t>(exts));
    rand_to<mpmode>(out);
    return out;
}

} // namespace random
} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/rand.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/uniform.hpp
/**
 * @file
 * @brief Uniform random distribution utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace random {

/**
 * @brief Fill an output tensor with uniform random values in [low, high)
 *        (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Output mdspan, mdarray, scalar, etc.
 * @param low Lower bound of the uniform distribution (inclusive).
 * @param high Upper bound of the uniform distribution (exclusive).
 *
 * @note The underlying generator produces values in [0, 1), which are then
 *       scaled and shifted to the target range.
 * @note If high == low, the output becomes constant equal to low.
 * @note Behavior for high < low follows the affine transform and is not
 *       explicitly validated.
 *
 * @see mdtensor::random::uniform for the out-of-place version that allocates an
 *      output and fills it.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t>
inline constexpr void uniform_to(in_t &&in, const double &low = 0,
                                 const double &high = 1) noexcept {
    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));

    using T = typename decltype(in_mds)::value_type;

    random::rand_to(in_mds);

    if constexpr (mpmode == MPMode::SIMD) {
        multiply_to<mpmode>(in_mds, static_cast<const T>(high - low), in_mds);
        add_to<mpmode>(in_mds, static_cast<const T>(low), in_mds);
        return;

    } else {
        core::batch<mpmode>(
            [&](auto &&in) {
                in() = static_cast<const T>(high - low) * in() +
                       static_cast<const T>(low);
            },
            in_mds);
    }
}

/**
 * @brief Create an array of uniform random values in [low, high)
 *        (out-of-place).
 *
 * @tparam dtype (optional) Floating-point element type. Default is float.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam exts_t (optional) Extents type. Default is extents<uint8_t>.
 *
 * @param exts Output extents.
 * @param low Lower bound of the uniform distribution (inclusive).
 * @param high Upper bound of the uniform distribution (exclusive).
 *
 * @return Newly allocated mdarray filled with uniform random values in [low,
 *         high).
 *
 * @see mdtensor::random::uniform_to for the in-place version that fills an
 *      existing output.
 */
template <std::floating_point dtype = float, MPMode mpmode = MPMode::NONE,
          extents_c exts_t = extents<uint8_t>>
[[nodiscard]] inline constexpr auto uniform(exts_t &&exts = exts_t{},
                                            const double &low = 0,
                                            const double &high = 1) noexcept {
    auto out = empty<dtype>(std::forward<exts_t>(exts));
    uniform_to<mpmode>(out, low, high);
    return out;
}

} // namespace random
} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/uniform.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/random.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/util/util.hpp
/**
 * @file
 * @brief String utilities header aggregator for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/util/to_string.hpp
/**
 * @file
 * @brief String conversion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <typename in_t>
[[nodiscard]] inline std::string to_string_impl(in_t &&in) {
    using index_t = typename std::remove_cvref_t<in_t>::index_type;

    std::string str = "[";

    if constexpr (in_t::rank() == 0) {
        return std::to_string(in());

    } else {
        for (index_t i = 0; i < in.extent(0); i++) {
            str += to_string_impl(core::submdspan_from_left(in, i));

            if (i < in.extent(0) - 1) {
                str += ", ";
            }
        }
    }

    return str + "]";
}

} // namespace detail

/**
 * @brief Convert an extents object into a shape string.
 *
 * @tparam in_t Extents type satisfying extents_c.
 *
 * @param in Input extents.
 *
 * @return Shape string in the form "(d0, d1, ...)".
 *
 * @note This overload is useful for printing tensor shapes.
 */
template <extents_c exts_t>
[[nodiscard]] inline std::string to_string(exts_t &&exts) {
    using exts_base_t = std::remove_cvref_t<exts_t>;

    std::string str = "(";

    for (size_t i = 0; i < exts_base_t::rank(); i++) {
        str += std::to_string(exts.extent(i));
        if (i < exts_base_t::rank() - 1) {
            str += ", ";
        }
    }

    return str + ")";
}

/**
 * @brief Convert a tensor-like object into a nested bracket string.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return Nested bracket string representation.
 *
 * @note The input is converted to a const mdspan view before formatting.
 *
 * @see mdtensor::to_string(const extents_c&) for shape formatting.
 */
template <typename in_t> [[nodiscard]] inline std::string to_string(in_t &&in) {
    return detail::to_string_impl(
        core::to_const_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/util/to_string.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/util/util.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/mdtensor.hpp
#endif // MDTENSOR_SINGLE_HEADER_INCLUDE_GUARD_

