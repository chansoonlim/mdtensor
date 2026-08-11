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


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/core.hpp
/**
 * @file
 * @brief Core utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/base.hpp
/**
 * @file
 * @brief Base utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/common.hpp
/**
 * @file
 * @brief Common utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/core.hpp
/**
 * @file
 * @brief Core utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// TODO: Remove when C++23 std::mdspan supports
#ifndef MDSPAN_SINGLE_HEADER_INCLUDE_GUARD_ // for godbolt test
#include <experimental/mdarray>
#include <experimental/mdspan>
#endif

namespace mdtensor {

#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
#define REAL_GCC __GNUC__ // probably
#endif

// TODO: modify under define
#if defined(_OPENMP) && defined(REAL_GCC)
#define MDTENSOR_USE_OPENMP
#endif

namespace core {
namespace stdex = std::experimental;
} // namespace core

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/core.hpp

namespace mdtensor::core {

template <std::integral index_t>
[[nodiscard]] constexpr index_t bounding_index(index_t index,
                                               const std::size_t &bound) {
    if constexpr (std::is_signed_v<index_t>) {
        if (index < index_t{0}) {
            index = static_cast<index_t>(bound + 1 -
                                         static_cast<std::size_t>(-index));
        }
    }

    if (index < index_t{0} || bound < static_cast<std::size_t>(index)) {
        throw std::out_of_range(
            "Index is out of bounds: " + std::to_string(index) +
            " is not in [0, " + std::to_string(bound) + ").");
    }

    return index;
}

template <std::integral in_t, in_t... ins, typename compare_t>
[[nodiscard]] consteval auto
get_sorted_array(std::integer_sequence<in_t, ins...>,
                 compare_t compare) noexcept {
    auto arr = std::array{ins...};
    std::sort(arr.begin(), arr.end(), compare);
    return arr;
}

template <std::size_t rank, std::integral axes_t, axes_t... axes,
          typename compare_t>
[[nodiscard]] consteval auto
get_sorted_axes(std::integer_sequence<axes_t, axes...>,
                compare_t compare) noexcept {
    constexpr auto arr =
        get_sorted_array(std::index_sequence<static_cast<std::size_t>(
                             bounding_index<axes_t>(axes, rank - 1))...>{},
                         compare);

    if constexpr (1 < arr.size()) {
        static_assert(
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return ((arr[Is] != arr[Is + 1]) && ...);
            }(std::make_index_sequence<arr.size() - 1>{}),
            "Duplicate axes are not allowed.");
    }

    return arr;
}

template <typename value_t, std::size_t size>
[[nodiscard]] constexpr bool contains(const std::array<value_t, size> &array,
                                      const value_t &value) noexcept {
    for (const auto element : array) {
        if (element == value) {
            return true;
        }
    }

    return false;
}

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/common.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/concept.hpp
/**
 * @file
 * @brief Concept utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor::core {

template <typename T>
concept integral_c = std::integral<std::remove_cvref_t<T>>;

template <typename T>
concept unsigned_integral_c = std::unsigned_integral<std::remove_cvref_t<T>>;

template <typename T>
concept floating_point_c = std::floating_point<std::remove_cvref_t<T>>;

template <typename T>
concept arithmetic_c = std::integral<std::remove_cvref_t<T>> ||
                       std::floating_point<std::remove_cvref_t<T>>;

namespace detail {

template <typename T> struct is_nullopt_impl : std::false_type {};

template <> struct is_nullopt_impl<std::nullopt_t> : std::true_type {};

} // namespace detail

template <typename T> struct is_nullopt_t : detail::is_nullopt_impl<T> {};

template <typename T> constexpr bool is_nullopt_t_v = is_nullopt_t<T>::value;

template <typename T>
concept nullopt_t_c = is_nullopt_t_v<std::remove_cvref_t<T>>;

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/concept.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/extents.hpp
/**
 * @file
 * @brief Extents utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor::core {

template <typename IndexType, std::size_t... Extents>
using extents = stdex::extents<IndexType, Extents...>;

template <typename IndexType, std::size_t Rank>
using dextents = stdex::dextents<IndexType, Rank>;

// NOTE: dims will be included in C++23
// (https://en.cppreference.com/w/cpp/container/mdspan/extents)
template <std::size_t Rank, class IndexType = std::size_t>
using dims = dextents<IndexType, Rank>;

namespace detail {

template <typename T> struct is_extents_impl : std::false_type {};

template <typename IndexType, std::size_t... Extents>
struct is_extents_impl<extents<IndexType, Extents...>> : std::true_type {};

} // namespace detail

// NOTE: stdex::detail::__is_extents is not used for godbolt test compatibility
template <typename T> struct is_extents : detail::is_extents_impl<T> {};

template <typename T> constexpr bool is_extents_v = is_extents<T>::value;

template <typename T>
concept extents_c = is_extents_v<std::remove_cvref_t<T>>;

constexpr auto dynamic_extent = stdex::dynamic_extent;
constexpr auto dyn = dynamic_extent;

namespace detail {

template <std::size_t Size> struct signed_by_size;

template <> struct signed_by_size<1> {
    using type = std::int8_t;
};

template <> struct signed_by_size<2> {
    using type = std::int16_t;
};

template <> struct signed_by_size<4> {
    using type = std::int32_t;
};

template <> struct signed_by_size<8> {
    using type = std::int64_t;
};

template <std::size_t Size>
using signed_by_size_t = typename signed_by_size<Size>::type;

template <typename T>
constexpr bool valid_extent_index_v =
    integral_c<T> && !std::same_as<std::remove_cvref_t<T>, bool>;

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
    requires(sizeof...(Ts) > 0 &&
             requires { typename common_index_type_impl<T1, T2>::type; })
struct common_index_type_impl<T1, T2, Ts...> {
  public:
    using type = typename common_index_type_impl<
        typename common_index_type_impl<T1, T2>::type, Ts...>::type;
};

} // namespace detail

template <typename... Ts>
using common_index_type_t =
    typename detail::common_index_type_impl<Ts...>::type;

[[nodiscard]] constexpr auto to_extents(auto &&shape) {
    using base_t = std::remove_cvref_t<decltype(shape)>;

    if constexpr (extents_c<base_t>) {
        // If the input is already an extents, just return it as-is
        return std::forward<decltype(shape)>(shape);

    } else if constexpr (integral_c<base_t>) {
        if (shape < base_t{0}) {
            throw std::invalid_argument("shape must be non-negative");
        }

        using index_t = std::make_unsigned_t<base_t>;

        return dextents<index_t, 1>{static_cast<index_t>(shape)};
    }
}

template <extents_c in_t>
[[nodiscard]] constexpr std::size_t extents_size(in_t &&in) noexcept {
    if constexpr (in.rank() == 0) {
        // NOTE: mdspan with rank 0 can capture a single element.
        return 1;

    } else if constexpr (in.rank_dynamic() == 0) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return (in.static_extent(Is) * ...);
        }(std::make_index_sequence<in.rank()>{});

    } else {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return (static_cast<std::size_t>(in.extent(Is)) * ...);
        }(std::make_index_sequence<in.rank()>{});
    }
}

template <extents_c in_t>
[[nodiscard]] consteval bool is_always_same_extents() noexcept {
    return true;
}

template <extents_c in1_t, extents_c in2_t, extents_c... ins_t>
[[nodiscard]] consteval bool is_always_same_extents() noexcept {
    using base1_t = std::remove_cvref_t<in1_t>;
    using base2_t = std::remove_cvref_t<in2_t>;

    if constexpr (base1_t::rank() != base2_t::rank()) {
        return false;

    } else if constexpr (base1_t::rank_dynamic() != 0 ||
                         base2_t::rank_dynamic() != 0) {
        return false;

    } else if constexpr ([&]<std::size_t... Is>(std::index_sequence<Is...>) {
                             return ((base1_t::static_extent(Is) !=
                                      base2_t::static_extent(Is)) ||
                                     ...);
                         }(std::make_index_sequence<base1_t::rank()>{})) {
        return false;
    }

    if constexpr (sizeof...(ins_t) != 0) {
        return is_always_same_extents<in2_t, ins_t...>();

    } else {
        return true;
    }
}

template <extents_c in_t>
[[nodiscard]] constexpr bool is_same_extents(in_t &&in) noexcept {
    return true;
}

template <extents_c in1_t, extents_c in2_t, extents_c... ins_t>
[[nodiscard]] constexpr bool is_same_extents(in1_t &&in1, in2_t &&in2,
                                             ins_t &&...ins) noexcept {
    // NOTE: direct comparison of extents cannot catch rank difference.

    using base1_t = std::remove_cvref_t<in1_t>;
    using base2_t = std::remove_cvref_t<in2_t>;

    if constexpr (base1_t::rank() != base2_t::rank()) {
        return false;
    }

    using index_t = common_index_type_t<typename base1_t::index_type,
                                        typename base2_t::index_type>;

    for (std::size_t i = 0; i < base1_t::rank(); i++) {
        if (static_cast<index_t>(in1.extent(i)) !=
            static_cast<index_t>(in2.extent(i))) {
            return false;
        }
    }

    if constexpr (sizeof...(ins_t) != 0) {
        return is_same_extents(in2, ins...);

    } else {
        return true;
    }
}

template <std::size_t offset, std::size_t rank, extents_c in_t>
[[nodiscard]] constexpr auto slice_extents(in_t &&in) noexcept {
    using index_t = typename std::remove_cvref_t<in_t>::index_type;

    static_assert(in.rank() >= offset + rank,
                  "Incompatible offset and rank for slicing.");

    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return extents<index_t, in.static_extent(offset + Is)...>{
            in.extent(offset + Is)...};
    }(std::make_index_sequence<rank>{});
}

template <std::size_t rank, extents_c in_t>
[[nodiscard]] constexpr auto slice_extents_from_left(in_t &&in) noexcept {
    return slice_extents<0, rank>(std::forward<in_t>(in));
}

template <std::size_t rank, extents_c in_t>
[[nodiscard]] constexpr auto slice_extents_from_right(in_t &&in) noexcept {
    return slice_extents<std::remove_cvref_t<in_t>::rank() - rank, rank>(
        std::forward<in_t>(in));
}

template <extents_c in1_t, extents_c in2_t, extents_c... ins_t>
[[nodiscard]] constexpr auto compose_extents(in1_t &&in1, in2_t &&in2,
                                             ins_t &&...ins) noexcept {
    using base1_t = std::remove_cvref_t<in1_t>;
    using base2_t = std::remove_cvref_t<in2_t>;
    using index_t = common_index_type_t<typename base1_t::index_type,
                                        typename base2_t::index_type>;

    const auto cexts =
        [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>,
                                                  std::index_sequence<Js...>) {
            return extents<index_t, base1_t::static_extent(Is)...,
                           base2_t::static_extent(Js)...>{
                static_cast<index_t>(in1.extent(Is))...,
                static_cast<index_t>(in2.extent(Js))...};
        }(std::make_index_sequence<base1_t::rank()>{},
          std::make_index_sequence<base2_t::rank()>{});

    if constexpr (sizeof...(ins_t) == 0) {
        return cexts;

    } else {
        return compose_extents(cexts, std::forward<ins_t>(ins)...);
    }
}

namespace {

template <extents_c in_t>
[[nodiscard]] constexpr auto expand_extents_dims_impl_(in_t &&in,
                                                       std::index_sequence<>) {
    return std::forward<in_t>(in);
}

template <extents_c in_t, std::size_t axis, std::size_t... axes>
[[nodiscard]] constexpr auto
expand_extents_dims_impl_(in_t &&in, std::index_sequence<axis, axes...>) {
    // NOTE: this function requires ordered axes and does not check duplicates.

    using base_t = std::remove_cvref_t<in_t>;

    return expand_extents_dims_impl_(
        compose_extents(slice_extents_from_left<axis>(std::forward<in_t>(in)),
                        extents<uint8_t, 1>{},
                        slice_extents_from_right<base_t::rank() - axis>(
                            std::forward<in_t>(in))),
        std::index_sequence<axes...>{});
}

} // namespace

template <extents_c in_t, std::integral axes_t, axes_t... axes>
[[nodiscard]] constexpr auto
expand_extents_dims(in_t &&in, std::integer_sequence<axes_t, axes...>) {
    constexpr auto axes_sorted = get_sorted_axes<in.rank() + sizeof...(axes)>(
        std::integer_sequence<axes_t, axes...>{}, std::less<std::size_t>{});

    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return expand_extents_dims_impl_(
            std::forward<in_t>(in), std::index_sequence<axes_sorted[Is]...>{});
    }(std::make_index_sequence<axes_sorted.size()>{});
}

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/extents.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/mdarray.hpp
/**
 * @file
 * @brief Mdarray utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor::core {

template <typename ElementType, typename Extents,
          typename LayoutPolicy = stdex::layout_right,
          typename Container = std::vector<ElementType>>
using mdarray = stdex::mdarray<ElementType, Extents, LayoutPolicy, Container>;

namespace detail {

template <typename T> struct is_mdarray_impl : std::false_type {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename ContainerType>
struct is_mdarray_impl<
    mdarray<ElementType, ExtentsType, LayoutType, ContainerType>>
    : std::true_type {};

} // namespace detail

template <typename T> struct is_mdarray : detail::is_mdarray_impl<T> {};

template <typename T> constexpr bool is_mdarray_v = is_mdarray<T>::value;

template <typename T>
concept mdarray_c = is_mdarray_v<std::remove_cvref_t<T>>;

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/mdarray.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/mdspan.hpp
/**
 * @file
 * @brief Mdspan utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor::core {

template <typename ElementType, typename Extents,
          typename LayoutPolicy = stdex::layout_right,
          typename AccessorPolicy = stdex::default_accessor<ElementType>>
using mdspan =
    stdex::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

namespace detail {

template <typename T> struct is_mdspan_impl : std::false_type {};

template <typename ElementType, typename ExtentsType, typename LayoutType,
          typename AccessorType>
struct is_mdspan_impl<
    mdspan<ElementType, ExtentsType, LayoutType, AccessorType>>
    : std::true_type {};

} // namespace detail

template <typename T> struct is_mdspan : detail::is_mdspan_impl<T> {};

template <typename T> constexpr bool is_mdspan_v = is_mdspan<T>::value;

template <typename T>
concept mdspan_c = is_mdspan_v<std::remove_cvref_t<T>>;

[[nodiscard]] constexpr auto to_mdspan(auto &&io) {
    if constexpr (mdspan_c<decltype(io)>) {
        // If the input is already an mdspan, just return it as-is
        return std::forward<decltype(io)>(io);

    } else if constexpr (requires { io.to_mdspan(); }) {
        // If the input has a to_mdspan() member function, call it
        return io.to_mdspan();

    } else {
        // If the input is not an mdspan,
        // create a new mdspan that points to the input data
        using element_t = std::remove_reference_t<decltype(io)>;
        using extents_t = extents<std::uint8_t>;

        return mdspan<element_t, extents_t>{std::addressof(io), extents_t{}};
    }
}

[[nodiscard]] constexpr auto to_const_mdspan(auto &&in) {
    if constexpr (mdspan_c<decltype(in)>) {
        if constexpr (std::is_const_v<typename std::remove_reference_t<
                          decltype(in)>::element_type>) {
            // If the input is already a const mdspan, just return it as-is
            return std::forward<decltype(in)>(in);

        } else {
            // If the input is a non-const mdspan,
            // create a new const mdspan with the same data handle and mapping
            using in_t = std::remove_cvref_t<decltype(in)>;

            return mdspan<
                const typename in_t::value_type, typename in_t::extents_type,
                typename in_t::layout_type,
                stdex::default_accessor<const typename in_t::value_type>>(
                in.data_handle(), in.mapping(), {});
        }

    } else {
        // If the input is not an mdspan,
        // create a new const mdspan with the same data handle and mapping
        auto mds = to_mdspan(std::forward<decltype(in)>(in));

        using mds_t = std::remove_cvref_t<decltype(mds)>;

        return mdspan<
            const typename mds_t::value_type, typename mds_t::extents_type,
            typename mds_t::layout_type,
            stdex::default_accessor<const typename mds_t::value_type>>(
            mds.data_handle(), mds.mapping(), {});
    }
}

[[nodiscard]] constexpr auto to_output_mdspan(auto &&out) {
    if constexpr (mdspan_c<decltype(out)>) {
        static_assert(
            !std::is_const_v<
                typename std::remove_reference_t<decltype(out)>::element_type>,
            "Output mdspan must not be const");

        // If the output is already an mdspan, just return it as-is
        return std::forward<decltype(out)>(out);

    } else {
        static_assert(std::is_lvalue_reference_v<decltype(out)>,
                      "Output owners and scalars must be passed as lvalues.");

        return to_mdspan(std::forward<decltype(out)>(out));
    }
}

template <typename T>
using to_mdspan_t = decltype(to_mdspan(std::declval<T>()));

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/mdspan.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/submdspan.hpp
/**
 * @file
 * @brief Submdspan utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor::core {

constexpr auto full_extent = stdex::full_extent;

template <std::size_t start, std::size_t end>
using slice =
    stdex::strided_slice<std::integral_constant<std::size_t, start>,
                         std::integral_constant<std::size_t, end - start>,
                         std::integral_constant<std::size_t, 1>>;

[[nodiscard]] constexpr auto submdspan(auto &&io, auto &&...slices) {
    return stdex::submdspan(to_mdspan(std::forward<decltype(io)>(io)),
                            std::forward<decltype(slices)>(slices)...);
}

template <std::size_t lspace = 0, std::size_t rspace = 0>
[[nodiscard]] constexpr auto submdspan_with_space(auto &&io, auto &&...slices) {
    return [&]<std::size_t... Is, std::size_t... Js>(
               std::index_sequence<Is...>, std::index_sequence<Js...>) {
        return submdspan(to_mdspan(std::forward<decltype(io)>(io)),
                         ((void)Is, full_extent)...,
                         std::forward<decltype(slices)>(slices)...,
                         ((void)Js, full_extent)...);
    }(std::make_index_sequence<lspace>{}, std::make_index_sequence<rspace>{});
}

template <std::size_t lspace = 0>
[[nodiscard]] constexpr auto submdspan_from_left(auto &&io, auto &&...slices) {
    using base_t = std::remove_reference_t<decltype(io)>;

    constexpr std::size_t rspace =
        to_mdspan_t<base_t>::rank() - (lspace + sizeof...(slices));

    return submdspan_with_space<lspace, rspace>(
        std::forward<decltype(io)>(io),
        std::forward<decltype(slices)>(slices)...);
}

template <std::size_t rspace = 0>
[[nodiscard]] constexpr auto submdspan_from_right(auto &&io, auto &&...slices) {
    using base_t = std::remove_reference_t<decltype(io)>;

    constexpr std::size_t lspace =
        to_mdspan_t<base_t>::rank() - (rspace + sizeof...(slices));

    return submdspan_with_space<lspace, rspace>(
        std::forward<decltype(io)>(io),
        std::forward<decltype(slices)>(slices)...);
}

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/submdspan.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/base/base.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/broadcast/broadcast.hpp
/**
 * @file
 * @brief Broadcast utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/broadcast/bcast.hpp
/**
 * @file
 * @brief Broadcast utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/broadcast/bcast_extents.hpp
/**
 * @file
 * @brief Broadcast extents utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor::core {
namespace detail {

template <std::size_t I, std::size_t brank, extents_c in_t>
[[nodiscard]] consteval std::size_t aligned_static_extent() noexcept {
    using base_t = std::remove_cvref_t<in_t>;

    constexpr std::size_t rank = base_t::rank();

    static_assert(I < brank, "Index I must be less than broadcast rank brank.");
    static_assert(rank <= brank,
                  "Input rank must be less than or equal to broadcast rank.");

    if constexpr (I < brank - rank) {
        return 1;

    } else {
        return base_t::static_extent(I - (brank - rank));
    }
}

template <std::size_t... Extents>
[[nodiscard]] consteval std::size_t broadcast_static_extent() noexcept {
    static_assert(sizeof...(Extents) > 0,
                  "At least one extent must be provided for broadcasting.");

    if constexpr (((Extents == 1 || Extents == dyn) && ...)) {
        // return dyn if any extent is dyn, else return 1
        return std::max({Extents...});

    } else {
        // select the extent that is not 1 or dyn
        constexpr std::size_t bext =
            std::max({((Extents != 1 && Extents != dyn) ? Extents : 0)...});

        static_assert(
            ((Extents == bext || Extents == 1 || Extents == dyn) && ...),
            "Incompatible static extents for broadcasting.");

        return bext;
    }
}

template <std::size_t I, std::size_t brank, extents_c in_t>
[[nodiscard]] constexpr auto aligned_extent(in_t &&in) noexcept {
    using index_t = typename std::remove_cvref_t<in_t>::index_type;

    constexpr std::size_t rank = in.rank();

    static_assert(I < brank, "Index I must be less than broadcast rank brank.");
    static_assert(rank <= brank,
                  "Input rank must be less than or equal to broadcast rank.");

    if constexpr (I < brank - rank) {
        return index_t{1};

    } else {
        return in.extent(I - (brank - rank));
    }
}

template <typename index_t, std::convertible_to<index_t>... exts_t>
[[nodiscard]] constexpr index_t broadcast_extent(exts_t &&...exts) {
    static_assert(sizeof...(exts) > 0,
                  "At least one extent must be provided for broadcasting.");

    index_t bext = 1;

    for (const index_t &ext : {static_cast<index_t>(exts)...}) {
        if (ext == 1) {
            continue;

        } else if (bext == 1) {
            bext = ext;

        } else if (ext != bext) {
            throw std::invalid_argument(
                "Incompatible extents for broadcasting.");
        }
    }

    return bext;
}

} // namespace detail

template <extents_c... ins_t>
[[nodiscard]] constexpr auto broadcast_extents(ins_t &&...ins) {
    static_assert(sizeof...(ins) > 0,
                  "At least one extents must be provided for broadcasting.");

    using index_t =
        common_index_type_t<typename std::remove_cvref_t<ins_t>::index_type...>;

    constexpr std::size_t brank = std::max({ins.rank()...});

    if constexpr (brank == 0) {
        return extents<index_t>{};

    } else {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            const auto static_extent_at = [&]<std::size_t I>() {
                return detail::broadcast_static_extent<
                    detail::aligned_static_extent<I, brank, ins_t>()...>();
            };

            const auto extent_at = [&]<std::size_t I>() {
                return detail::broadcast_extent<index_t>(
                    detail::aligned_extent<I, brank>(
                        std::forward<ins_t>(ins))...);
            };

            return extents<index_t,
                           static_extent_at.template operator()<Is>()...> {
                extent_at.template operator()<Is>()...
            };
        }(std::make_index_sequence<brank>{});
    }
}

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/broadcast/bcast_extents.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/broadcast/bcast_to.hpp
/**
 * @file
 * @brief Broadcast utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor::core {

[[nodiscard]] constexpr auto broadcast_to(auto &&in, auto &&shape) {
    const auto in_mds = to_const_mdspan(std::forward<decltype(in)>(in));
    const auto exts = to_extents(std::forward<decltype(shape)>(shape));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using exts_t = std::remove_cvref_t<decltype(exts)>;

    constexpr std::size_t org_rank = in_mds_t::rank();
    constexpr std::size_t new_rank = exts_t::rank();

    static_assert(org_rank <= new_rank, "Incompatible ranks for broadcasting.");

    if constexpr (is_always_same_extents<typename in_mds_t::extents_type,
                                         exts_t>()) {
        return in_mds; // change to const mdspan

    } else if constexpr (org_rank == 0) {
        using index_t = typename exts_t::index_type;

        auto new_strides = std::array<index_t, new_rank>{};

        for (std::size_t i = 0; i < new_rank; i++) {
            new_strides[i] = 0;
        }

        return mdspan<typename in_mds_t::element_type, exts_t,
                      stdex::layout_stride, typename in_mds_t::accessor_type>{
            in_mds.data_handle(),
            stdex::layout_stride::mapping{exts, new_strides}};

    } else {
        using index_t = typename exts_t::index_type;
        using cindex_t =
            common_index_type_t<typename in_mds_t::index_type, index_t>;

        // ni = new_rank - org_rank + oi
        const auto get_ni = [](std::size_t i) {
            return new_rank - org_rank + i;
        };

        // assertion
        static_assert(
            [&] {
                for (std::size_t i = 0; i < org_rank; i++) {
                    const auto src = in_mds_t::static_extent(i);
                    const auto dst = exts_t::static_extent(get_ni(i));

                    if (src != dyn && dst != dyn && src != dst && src != 1) {
                        return false;
                    }
                }
                return true;
            }(),
            "Incompatible extents for broadcasting.");

        for (std::size_t i = 0; i < org_rank; i++) {
            if (static_cast<cindex_t>(in_mds.extent(i)) !=
                    static_cast<cindex_t>(exts.extent(get_ni(i))) &&
                static_cast<cindex_t>(in_mds.extent(i)) != cindex_t{1}) {
                throw std::invalid_argument(
                    "Incompatible extents for broadcasting.");
            }
        }

        // calculation
        auto new_strides = std::array<index_t, new_rank>{};

        for (std::size_t i = 0; i < new_rank - org_rank; i++) {
            new_strides[i] = 0;
        }

        for (std::size_t i = 0; i < org_rank; i++) {
            if (static_cast<cindex_t>(in_mds.extent(i)) ==
                static_cast<cindex_t>(exts.extent(get_ni(i)))) {
                new_strides[get_ni(i)] = static_cast<index_t>(in_mds.stride(i));

            } else {
                new_strides[get_ni(i)] = 0;
            }
        }

        return mdspan<typename in_mds_t::element_type, exts_t,
                      stdex::layout_stride, typename in_mds_t::accessor_type>{
            in_mds.data_handle(),
            stdex::layout_stride::mapping{exts, new_strides}};
    }
}

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/broadcast/bcast_to.hpp

namespace mdtensor::core {
namespace detail {

template <std::size_t... uranks, mdspan_c... ios_t>
[[nodiscard]] constexpr auto
get_broadcast_extents(std::index_sequence<uranks...>, ios_t &&...ios) {
    static_assert(sizeof...(uranks) == sizeof...(ios_t),
                  "Number of uranks must match number of inputs.");
    static_assert(((ios.rank() >= uranks) && ...),
                  "Input rank must be greater than or equal to urank.");

    return broadcast_extents(
        slice_extents_from_left<ios.rank() - uranks>(ios.extents())...);
}

} // namespace detail

template <std::size_t... uranks, bool... bcast>
[[nodiscard]] constexpr auto broadcast(std::index_sequence<uranks...>,
                                       std::integer_sequence<bool, bcast...>,
                                       auto &&...ios) {
    static_assert(sizeof...(uranks) == sizeof...(ios));
    static_assert(sizeof...(bcast) == sizeof...(ios));

    // calculate mdspans for inputs and outputs
    const auto ios_mds =
        std::make_tuple(to_mdspan(std::forward<decltype(ios)>(ios))...);

    // calculate broadcasted extents
    constexpr auto ur = std::array{uranks...};

    const auto bexts = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return detail::get_broadcast_extents(std::index_sequence<ur[Is]...>{},
                                             std::get<Is>(ios_mds)...);
    }(std::make_index_sequence<sizeof...(ios)>{});

    // calculate broadcasted mdspans
    constexpr auto bc = std::array{bcast...};

    return std::make_tuple(
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            const auto get_broadcasted = [&]<std::size_t I>() {
                if constexpr (!bc[I]) {
                    // change to mdspan without broadcasting
                    return std::get<I>(ios_mds);

                } else if constexpr (bexts.rank() == 0) {
                    // change to const mdspan without broadcasting
                    return to_const_mdspan(std::get<I>(ios_mds));

                } else {
                    // broadcast to const mdspan
                    return broadcast_to(
                        std::get<I>(ios_mds),
                        compose_extents(bexts,
                                        slice_extents_from_right<ur[I]>(
                                            std::get<I>(ios_mds).extents())));
                }
            };

            return std::make_tuple(
                get_broadcasted.template operator()<Is>()...);
        }(std::make_index_sequence<sizeof...(ios)>{}),
        bexts);
}

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/broadcast/bcast.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/broadcast/broadcast.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/manipulation/manipulation.hpp
/**
 * @file
 * @brief Manipulation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/manipulation/expand_dims.hpp
/**
 * @file
 * @brief Expand dimensions utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/manipulation/reshape.hpp
/**
 * @file
 * @brief Reshape utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/tensor/tensor.hpp
/**
 * @file
 * @brief Tensor utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/tensor/make_broadcasted_tensor.hpp
/**
 * @file
 * @brief Output utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/tensor/type.hpp
/**
 * @file
 * @brief Tensor type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor::core {

using bool_value_t = std::int8_t;

template <typename value_t, extents_c extent_t>
using tensor = std::conditional_t<
    extent_t::rank() == 0, value_t,
    std::conditional_t<
        extent_t::rank_dynamic() == 0,
        core::mdarray<value_t, extent_t, stdex::layout_right,
                      std::array<value_t, core::extents_size(extent_t{})>>,
        std::conditional_t<
            std::is_same_v<value_t, bool>,
            core::mdarray<bool_value_t, extent_t, stdex::layout_right,
                          std::vector<bool_value_t>>,
            core::mdarray<value_t, extent_t, stdex::layout_right,
                          std::vector<value_t>>>>>;

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/tensor/type.hpp

namespace mdtensor::core {
namespace detail {

template <typename T>
constexpr bool data_bool_v = std::same_as<std::remove_cvref_t<T>, bool>;

template <typename T> constexpr bool data_fpoint_v = floating_point_c<T>;

template <typename T>
constexpr bool data_integer_v =
    std::integral<std::remove_cvref_t<T>> && !data_bool_v<T>;

template <typename T>
constexpr bool data_scalar_v =
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

// floating + scalar -> floating
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
  private:
    using pair_t =
        typename common_data_pair_impl<std::remove_cvref_t<T1>,
                                       std::remove_cvref_t<T2>>::type;

  public:
    using type = typename data_promote_impl<pair_t, Ts...>::type;
};

} // namespace detail

template <typename... Ts>
    requires(sizeof...(Ts) > 0 && (detail::data_scalar_v<Ts> && ...))
using common_data_type_t = typename detail::data_promote_impl<Ts...>::type;

namespace detail {

template <typename dtype, typename... Ts> struct output_value {
    using type = dtype;
};

template <typename... Ts> struct output_value<void, Ts...> {
    using type = common_data_type_t<
        typename std::remove_cvref_t<to_mdspan_t<Ts>>::value_type...>;
};

template <typename dtype, typename Tuple> struct output_value_from_tuple;

template <typename dtype, typename... Ts>
struct output_value_from_tuple<dtype, std::tuple<Ts...>>
    : output_value<dtype, Ts...> {};

template <typename T> struct unwrap_optional {
    using type = std::remove_cvref_t<T>;
};

template <typename T> struct unwrap_optional<std::optional<T>> {
    using type = std::remove_cvref_t<T>;
};

template <typename... Ts> struct filter_nullopt;

template <> struct filter_nullopt<> {
    using type = std::tuple<>;
};

template <typename T, typename... Ts> struct filter_nullopt<T, Ts...> {
  private:
    using arg_t = typename unwrap_optional<std::remove_cvref_t<T>>::type;
    using tail_t = typename filter_nullopt<Ts...>::type;

  public:
    using type = std::conditional_t<
        std::same_as<std::remove_cvref_t<T>, std::nullopt_t>, tail_t,
        decltype(std::tuple_cat(std::declval<std::tuple<arg_t>>(),
                                std::declval<tail_t>()))>;
};

} // namespace detail

template <typename dtype, typename... Ts>
using output_value_t = typename detail::output_value<dtype, Ts...>::type;

template <typename dtype, typename... Ts>
using output_value_with_nullopt_t = typename detail::output_value_from_tuple<
    dtype, typename detail::filter_nullopt<Ts...>::type>::type;

template <typename dtype = void, std::size_t... uranks, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
make_broadcasted_tensor(std::index_sequence<uranks...>, uout_exts_t &&uout_exts,
                        auto &&...ins) {
    static_assert(sizeof...(uranks) == sizeof...(ins),
                  "Number of uranks must match number of inputs.");

    constexpr std::size_t ins_num = sizeof...(uranks);

    if constexpr (ins_num == 0) {
        return make_tensor<dtype>(std::forward<uout_exts_t>(uout_exts));

    } else {
        using value_t = output_value_t<dtype, decltype(ins)...>;

        // calculate broadcasted extents
        const auto bexts = detail::get_broadcast_extents(
            std::index_sequence<uranks...>{},
            to_const_mdspan(std::forward<decltype(ins)>(ins))...);

        return make_tensor<value_t>(
            compose_extents(bexts, std::forward<uout_exts_t>(uout_exts)));
    }
}

template <typename dtype = void, extents_c uout_exts_t>
[[nodiscard]] constexpr auto make_broadcasted_tensor(uout_exts_t &&uout_exts,
                                                     auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return make_broadcasted_tensor<dtype>(
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

template <typename dtype = void, std::size_t... uranks>
[[nodiscard]] constexpr auto
make_broadcasted_tensors(std::index_sequence<uranks...>, auto &&uout_exts_tuple,
                         auto &&...ins) {
    static_assert(sizeof...(uranks) == sizeof...(ins),
                  "Number of uranks must match number of inputs.");

    constexpr std::size_t ins_num = sizeof...(uranks);
    constexpr std::size_t outs_num =
        std::tuple_size_v<std::remove_cvref_t<decltype(uout_exts_tuple)>>;

    if constexpr (ins_num == 0) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::tuple{
                make_tensor<dtype>(std::get<Is>(uout_exts_tuple))...};
        }(std::make_index_sequence<outs_num>{});

    } else {
        using value_t = output_value_t<dtype, decltype(ins)...>;

        // calculate broadcasted extents
        const auto bexts = detail::get_broadcast_extents(
            std::index_sequence<uranks...>{},
            to_const_mdspan(std::forward<decltype(ins)>(ins))...);

        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::tuple{make_tensor<value_t>(
                compose_extents(bexts, std::get<Is>(uout_exts_tuple)))...};
        }(std::make_index_sequence<outs_num>{});
    }
}

template <typename dtype = void>
[[nodiscard]] constexpr auto make_broadcasted_tensors(auto &&uout_exts_tuple,
                                                      auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return make_broadcasted_tensors<dtype>(
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<decltype(uout_exts_tuple)>(uout_exts_tuple),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

// TODO: develop make_reduce_outputs

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/tensor/make_broadcasted_tensor.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/tensor/make_reduced_tensor.hpp
/**
 * @file
 * @brief Output utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor::core {

template <typename dtype = void, bool keepdims = false, std::integral axes_t,
          axes_t... axes, std::size_t... uranks, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
make_reduced_tensor(std::integer_sequence<axes_t, axes...>,
                    std::index_sequence<uranks...>, uout_exts_t &&uout_exts,
                    auto &&...ins) {
    static_assert(sizeof...(ins) > 0, "At least one input must be provided.");
    static_assert(sizeof...(uranks) == sizeof...(ins),
                  "Number of uranks must match number of inputs.");

    // calculate input broadcasted extents
    const auto ins_bexts = detail::get_broadcast_extents(
        std::index_sequence<uranks...>{},
        to_const_mdspan(std::forward<decltype(ins)>(ins))...);

    using ins_bexts_t = decltype(ins_bexts);

    // get sorted array
    constexpr auto axes_sorted = [&]() {
        if constexpr (ins_bexts_t::rank() == 0) {
            return std::array<std::size_t, 0>{};

        } else if constexpr (sizeof...(axes) == 0) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::array{(ins_bexts_t::rank() - 1 - Is)...};
            }(std::make_index_sequence<ins_bexts_t::rank()>{});

        } else {
            return get_sorted_axes<ins_bexts_t::rank()>(
                std::integer_sequence<axes_t, axes...>{},
                std::greater<std::size_t>{});
        }
    }();

    // generate out_bexts
    constexpr auto not_axes_size = ins_bexts_t::rank() - axes_sorted.size();

    const auto out_bexts = [&]() {
        using index_t = typename ins_bexts_t::index_type;

        if constexpr (keepdims) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return extents<index_t,
                               (contains(axes_sorted, Is)
                                    ? std::size_t{1}
                                    : ins_bexts_t::static_extent(Is))...>{
                    (contains(axes_sorted, Is)
                         ? index_t{1}
                         : static_cast<index_t>(ins_bexts.extent(Is)))...};
            }(std::make_index_sequence<ins_bexts_t::rank()>{});

        } else {
            // generate unselected axes sequence
            constexpr auto not_axes_arr = [&]() {
                auto not_axes_arr = std::array<std::size_t, not_axes_size>{};

                std::size_t not_axes_idx = 0;
                for (std::size_t i = 0; i < ins_bexts_t::rank(); i++) {
                    if (!contains(axes_sorted, i)) {
                        not_axes_arr[not_axes_idx++] = i;
                    }
                }

                return not_axes_arr;
            }();

            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return extents<index_t,
                               ins_bexts_t::static_extent(not_axes_arr[Is])...>{
                    ins_bexts.extent(not_axes_arr[Is])...};
            }(std::make_index_sequence<not_axes_arr.size()>{});
        }
    }();

    // generate out
    using value_t = output_value_t<dtype, decltype(ins)...>;

    return make_tensor<value_t>(
        compose_extents(out_bexts, std::forward<uout_exts_t>(uout_exts)));
}

// TODO: develop make_reduce_outputs

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/tensor/make_reduced_tensor.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/tensor/make_tensor.hpp
/**
 * @file
 * @brief Make tensor utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor::core {

template <typename value_t = double, extents_c exts_t>
[[nodiscard]] constexpr auto make_tensor(exts_t &&exts) {
    using base_t = std::remove_cvref_t<decltype(exts)>;

    if constexpr (base_t::rank() == 0) {
        return tensor<value_t, base_t>{};

    } else {
        return tensor<value_t, base_t>{std::forward<exts_t>(exts)};
    }
}

[[nodiscard]] constexpr auto make_tensor_like(auto &&in) {
    const auto in_mds = to_const_mdspan(std::forward<decltype(in)>(in));

    using value_t = typename decltype(in_mds)::value_type;

    return make_tensor<value_t>(in_mds.extents());
}

template <typename T>
using make_tensor_like_t = decltype(make_tensor_like(std::declval<T>()));

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/tensor/make_tensor.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/tensor/tensor.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/ufunc/ufunc.hpp
/**
 * @file
 * @brief Ufunc utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/ufunc/batch.hpp
/**
 * @file
 * @brief Batch utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



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

template <Backend backend, std::size_t brank>
constexpr void batch(auto &&func, auto &&...ios) {
    // TODO: assert when backend is not specified in each funciton call
    // assert(backend != Backend::AUTO);
    [[maybe_unused]] constexpr auto be = [&]() {
        if constexpr (backend == Backend::AUTO) {
            return Backend::NATIVE; // temporary approach.

        } else {
            return backend;
        }
    }();

    if constexpr (
#ifdef MDTENSOR_USE_OPENMP
        be == Backend::OPENMP
#else
        false
#endif
    ) {
#ifdef MDTENSOR_USE_OPENMP
        detail::batch_impl_openmp<brank>(
            std::forward<decltype(func)>(func),
            to_mdspan(std::forward<decltype(ios)>(ios))...);
#endif

    } else {
        detail::batch_impl_native<brank>(
            std::forward<decltype(func)>(func),
            to_mdspan(std::forward<decltype(ios)>(ios))...);
    }
}

template <Backend backend, std::size_t... uranks, bool... bcast>
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

template <Backend backend, bool... bcast>
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
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/ufunc/batch.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/ufunc/reduce.hpp
/**
 * @file
 * @brief Reduce utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor::core {
namespace detail {

template <bool is_input, std::size_t axis, bool keepdims, mdspan_c io_t,
          typename index_t>
[[nodiscard]] constexpr auto reduce_input(io_t &&io, index_t i) {
    if constexpr (is_input) {
        return submdspan_from_left<axis>(std::forward<io_t>(io), i);

    } else if constexpr (keepdims) {
        return submdspan_from_left<axis>(std::forward<io_t>(io), 0);

    } else {
        return std::forward<io_t>(io);
    }
}

template <bool keepdims, extents_c bext_t, bool... is_input, mdspan_c... ios_t>
constexpr void batch_reduced(auto &&func, bext_t &&, std::index_sequence<>,
                             std::integer_sequence<bool, is_input...>,
                             ios_t &&...ios) {
    func(std::forward<ios_t>(ios)...);
}

template <bool keepdims, extents_c bext_t, std::size_t axis,
          std::size_t... axes, bool... is_input, mdspan_c... ios_t>
constexpr void
batch_reduced(auto &&func, bext_t &&bext, std::index_sequence<axis, axes...>,
              std::integer_sequence<bool, is_input...>, ios_t &&...ios) {
    static_assert(sizeof...(is_input) == sizeof...(ios_t));
    static_assert(((axis > axes) && ...), "Axes must be in descending order.");

    using index_t = typename std::remove_cvref_t<bext_t>::index_type;

    for (index_t i = 0; i < bext.extent(axis); i++) {
        batch_reduced<keepdims>(std::forward<decltype(func)>(func),
                                std::forward<decltype(bext)>(bext),
                                std::index_sequence<axes...>{},
                                std::integer_sequence<bool, is_input...>{},
                                reduce_input<is_input, axis, keepdims>(
                                    std::forward<ios_t>(ios), i)...);
    }
}

template <std::size_t... uranks, bool... is_input>
[[nodiscard]] constexpr auto
broadcast_only_input(std::index_sequence<uranks...>,
                     std::integer_sequence<bool, is_input...>, auto &&...ios) {
    static_assert(sizeof...(uranks) == sizeof...(ios));
    static_assert(sizeof...(is_input) == sizeof...(ios));

    // make helpers
    constexpr auto bin = std::array{is_input...};
    const auto ios_org =
        std::tuple{to_mdspan(std::forward<decltype(ios)>(ios))...};
    constexpr auto ios_num = std::tuple_size_v<decltype(ios_org)>;
    constexpr auto ios_uranks = std::array{uranks...};

    // separate inputs
    const auto ins_tuple = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        const auto get_input = [&]<std::size_t I>() {
            if constexpr (bin[I]) {
                return std::forward_as_tuple(std::get<I>(ios_org));

            } else {
                return std::forward_as_tuple();
            }
        };

        return std::tuple_cat(get_input.template operator()<Is>()...);
    }(std::make_index_sequence<sizeof...(ios)>{});

    constexpr auto ins_num = std::tuple_size_v<decltype(ins_tuple)>;

    static_assert(ins_num > 0, "At least one input must be provided.");

    constexpr auto ins_uranks = [&]() {
        auto ins_uranks = std::array<std::size_t, ins_num>{};
        std::size_t ins_idx = 0;
        for (std::size_t i = 0; i < ios_num; i++) {
            if (bin[i]) {
                ins_uranks[ins_idx++] = ios_uranks[i];
            }
        }
        return ins_uranks;
    }();

    // broadcast inputs only
    const auto [ins_bcast, ins_bexts] =
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return broadcast(std::index_sequence<ins_uranks[Is]...>{},
                             std::integer_sequence<bool, (void(Is), true)...>{},
                             std::get<Is>(ins_tuple)...);
        }(std::make_index_sequence<ins_num>{});

    // return broadcasted inputs and outputs in same order as original inputs
    return std::make_tuple(
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            const auto get_io = [&]<std::size_t I>() {
                constexpr std::size_t J =
                    [&]<std::size_t... Js>(std::index_sequence<Js...>) {
                        return (std::size_t{0} + ... +
                                static_cast<std::size_t>(bin[Js]));
                    }(std::make_index_sequence<I>{});

                if constexpr (bin[I]) {
                    // pass broadcasted input
                    return std::get<J>(ins_bcast);

                } else {
                    // pass through outputs
                    return std::get<I>(ios_org);
                }
            };

            return std::make_tuple(get_io.template operator()<Is>()...);
        }(std::make_index_sequence<sizeof...(ios)>{}),
        ins_bexts);
}

} // namespace detail

template <bool keepdims = false, std::integral axes_t, axes_t... axes,
          std::size_t... uranks, bool... is_input>
constexpr void reduce(auto &&func, std::integer_sequence<axes_t, axes...>,
                      std::index_sequence<uranks...>,
                      std::integer_sequence<bool, is_input...>, auto &&...ios) {
    // broadcast inputs only
    const auto [ios_bcast, ins_bexts] =
        detail::broadcast_only_input(std::index_sequence<uranks...>{},
                                     std::integer_sequence<bool, is_input...>{},
                                     std::forward<decltype(ios)>(ios)...);

    using ins_bexts_t = decltype(ins_bexts);

    // get sorted array
    constexpr auto axes_sorted = [&]() {
        if constexpr (ins_bexts_t::rank() == 0) {
            return std::array<std::size_t, 0>{};

        } else if constexpr (sizeof...(axes) == 0) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::array{(ins_bexts_t::rank() - 1 - Is)...};
            }(std::make_index_sequence<ins_bexts_t::rank()>{});

        } else {
            return get_sorted_axes<ins_bexts_t::rank()>(
                std::integer_sequence<axes_t, axes...>{},
                std::greater<std::size_t>{});
        }
    }();

    // batch
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        [&]<std::size_t... Js>(std::index_sequence<Js...>) {
            detail::batch_reduced<keepdims>(
                std::forward<decltype(func)>(func), ins_bexts,
                std::index_sequence<axes_sorted[Js]...>{},
                std::integer_sequence<bool, is_input...>{},
                std::get<Is>(ios_bcast)...);
        }(std::make_index_sequence<axes_sorted.size()>{});
    }(std::make_index_sequence<sizeof...(ios)>{});
}

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/ufunc/reduce.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/ufunc/ufunc.hpp

namespace mdtensor::core {

enum class Copy {
    TRUE,  // Copy the input tensor to a new tensor
    FALSE, // Do not copy the input tensor; return a view of the input tensor
    AUTO,  // Automatically determine whether to copy or not based on input
};

namespace detail {

template <typename in_t>
    requires(mdspan_c<in_t> || mdarray_c<in_t>)
[[nodiscard]] consteval bool is_always_c_contiguous() noexcept {
    return std::same_as<typename std::remove_cvref_t<in_t>::layout_type,
                        stdex::layout_right>;
}

template <mdspan_c in_t>
[[nodiscard]] constexpr bool is_c_contiguous(const in_t &in) noexcept {
    if constexpr (in.rank() == 0) {
        return true;

    } else {
        // Empty tensors have no observable element ordering.
        if (extents_size(in.extents()) == 0) {
            return true;
        }

        if (!in.is_unique() || !in.is_exhaustive() || !in.is_strided()) {
            return false;
        }

        std::size_t expected_stride = 1;

        for (std::size_t i = in.rank(); i-- > 0;) {
            const std::size_t extent = static_cast<std::size_t>(in.extent(i));

            if (extent > 1 &&
                static_cast<std::size_t>(in.stride(i)) != expected_stride) {
                return false;
            }

            expected_stride *= extent;
        }

        return true;
    }
}

template <extents_c exts_t>
[[nodiscard]] constexpr auto make_reshape_view(auto &&in, exts_t &&exts) {
    const auto in_mds = to_mdspan(std::forward<decltype(in)>(in));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using new_exts_t = std::remove_cvref_t<exts_t>;

    using out_mds_t =
        mdspan<typename in_mds_t::element_type, new_exts_t, stdex::layout_right,
               typename in_mds_t::accessor_type>;

    if constexpr (is_always_same_extents<typename in_mds_t::extents_type,
                                         new_exts_t>()) {
        return in_mds;

    } else {
        return out_mds_t{
            in_mds.data_handle(),
            typename out_mds_t::mapping_type{std::forward<exts_t>(exts)},
            in_mds.accessor()};
    }
}

template <extents_c exts_t>
[[nodiscard]] constexpr auto make_reshape_copy(auto &&in, exts_t &&exts) {
    const auto in_mds = to_const_mdspan(std::forward<decltype(in)>(in));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using value_t = typename in_mds_t::value_type;

    auto out = make_tensor<value_t>(std::forward<exts_t>(exts));

    batch<Backend::NATIVE, in_mds.rank()>(
        [&](auto &&in, auto &&out) { out() = in(); }, in_mds,
        make_reshape_view(out, in_mds.extents()));

    return out;
}

} // namespace detail

template <Copy copy = Copy::AUTO>
[[nodiscard]] constexpr auto reshape(auto &&in, auto &&shape) {
    const auto in_mds = to_mdspan(std::forward<decltype(in)>(in));
    const auto exts = to_extents(std::forward<decltype(shape)>(shape));

    using in_mds_t = std::remove_cvref_t<decltype(in_mds)>;
    using in_exts_t = typename in_mds_t::extents_type;
    using exts_t = std::remove_cvref_t<decltype(exts)>;

    if constexpr (in_exts_t::rank_dynamic() == 0 &&
                  exts_t::rank_dynamic() == 0) {
        static_assert(extents_size(in_exts_t{}) == extents_size(exts_t{}),
                      "Reshape error: input and output extents "
                      "must have the same size.");

    } else if (extents_size(in_mds.extents()) != extents_size(exts)) {
        throw std::invalid_argument("Reshape error: input and output "
                                    "extents must have the same size.");
    }

    constexpr bool can_borrow =
        std::is_lvalue_reference_v<decltype(in)> || mdspan_c<decltype(in)>;

    if constexpr (copy == Copy::AUTO) {
        if constexpr (can_borrow &&
                      detail::is_always_c_contiguous<in_mds_t>()) {
            return reshape<Copy::FALSE>(in_mds, exts);

        } else {
            return reshape<Copy::TRUE>(in_mds, exts);
        }

    } else if constexpr (copy == Copy::TRUE) {
        return detail::make_reshape_copy(in_mds, exts);

    } else if constexpr (copy == Copy::FALSE) {
        static_assert(can_borrow, "Reshape error: zero-copy reshape cannot "
                                  "bind to a temporary owning tensor.");

        if (!detail::is_c_contiguous(in_mds)) {
            throw std::invalid_argument("Reshape error: zero-copy reshape "
                                        "requires a C-contiguous input.");
        }

        if constexpr (is_always_same_extents<in_exts_t, exts_t>()) {
            return std::forward<decltype(in)>(in);

        } else {
            return detail::make_reshape_view(in_mds, exts);
        }
    }
}

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/manipulation/reshape.hpp

namespace mdtensor::core {

template <std::integral axes_t, axes_t... axes>
[[nodiscard]] constexpr auto
expand_dims(auto &&in, std::integer_sequence<axes_t, axes...>) {
    const auto in_mds = to_mdspan(std::forward<decltype(in)>(in));

    return reshape<Copy::FALSE>(
        in_mds, expand_extents_dims(in_mds.extents(),
                                    std::integer_sequence<axes_t, axes...>{}));
}

template <std::int64_t... axes>
[[nodiscard]] constexpr auto expand_dims(auto &&in) {
    return expand_dims(std::forward<decltype(in)>(in),
                       std::integer_sequence<std::int64_t, axes...>{});
}

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/manipulation/expand_dims.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/manipulation/manipulation.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/util/util.hpp
/**
 * @file
 * @brief Other utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/util/to_string.hpp
/**
 * @file
 * @brief To-string conversion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor::core {
namespace detail {

inline constexpr int numpy_default_precision = 8;

template <std::floating_point value_t>
[[nodiscard]] inline std::string floating_value_to_string(const value_t value) {
    if constexpr (requires {
                      { std::isnan(value) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(value)) {
            return "nan";
        }
    }

    if constexpr (requires {
                      { std::isinf(value) } -> std::convertible_to<bool>;
                  }) {
        if (std::isinf(value)) {
            return std::signbit(value) ? "-inf" : "inf";
        }
    }

    const value_t magnitude = std::abs(value);

    /*
     * NumPy-like scalar approximation:
     *
     * - fixed-point:      1e-4 <= |x| < 1e8, including zero
     * - scientific:       0 < |x| < 1e-4 or |x| >= 1e8
     *
     * NumPy's exact choice is array-wide, so mixed arrays may use
     * scientific notation even when an individual value would not.
     */
    const bool use_scientific =
        magnitude != value_t{0} &&
        (magnitude < value_t{1e-4} || magnitude >= value_t{1e8});

    const std::chars_format format = use_scientific
                                         ? std::chars_format::scientific
                                         : std::chars_format::fixed;

    std::array<char, 128> buffer{};

    // First obtain the shortest round-trippable representation.
    auto [ptr, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                   value, format);

    if (ec != std::errc{}) {
        throw std::runtime_error(
            "Failed to convert floating-point value to string.");
    }

    std::string text(buffer.data(), ptr);

    const auto fractional_digits = [](const std::string &str) -> std::size_t {
        const std::size_t exponent_pos = str.find_first_of("eE");

        const std::size_t mantissa_end =
            exponent_pos == std::string::npos ? str.size() : exponent_pos;

        const std::size_t decimal_pos = str.find('.');

        if (decimal_pos == std::string::npos || decimal_pos >= mantissa_end) {
            return 0;
        }

        return mantissa_end - decimal_pos - 1;
    };

    // NumPy default: at most 8 fractional digits.
    if (fractional_digits(text) >
        static_cast<std::size_t>(numpy_default_precision)) {
        auto [rounded_ptr, rounded_ec] =
            std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                          format, numpy_default_precision);

        if (rounded_ec != std::errc{}) {
            throw std::runtime_error(
                "Failed to convert floating-point value to string.");
        }

        text.assign(buffer.data(), rounded_ptr);
    }

    // Remove unnecessary trailing zeros, but preserve the decimal point.
    const std::size_t exponent_pos = text.find_first_of("eE");

    const std::size_t mantissa_end =
        exponent_pos == std::string::npos ? text.size() : exponent_pos;

    const std::size_t decimal_pos = text.find('.');

    if (decimal_pos == std::string::npos || decimal_pos >= mantissa_end) {
        // "21"     -> "21."
        // "1e+08"  -> "1.e+08"
        text.insert(mantissa_end, 1, '.');

    } else {
        std::size_t trim_end = mantissa_end;

        while (trim_end > decimal_pos + 1 && text[trim_end - 1] == '0') {
            --trim_end;
        }

        // Decimal point itself is intentionally retained.
        text.erase(trim_end, mantissa_end - trim_end);
    }

    return text;
}

template <arithmetic_c value_t>
[[nodiscard]] inline std::string value_to_string(const value_t &value) {
    using base_t = std::remove_cvref_t<value_t>;

    if constexpr (std::same_as<base_t, bool>) {
        return value ? "true" : "false";

    } else if constexpr (std::is_integral_v<base_t>) {
        if constexpr (std::is_signed_v<base_t>) {
            return std::to_string(static_cast<long long>(value));

        } else {
            return std::to_string(static_cast<unsigned long long>(value));
        }

    } else if constexpr (std::is_floating_point_v<base_t>) {
        return floating_value_to_string(value);
    }
}

template <typename value_t>
[[nodiscard]] inline std::string
value_to_string(const std::optional<value_t> &value) {
    if (!value.has_value()) {
        return "nullopt";
    }

    return value_to_string(*value);
}

} // namespace detail

template <extents_c exts_t>
[[nodiscard]] inline std::string to_string(exts_t &&exts) {
    using base_t = std::remove_cvref_t<exts_t>;

    std::string str = "(";

    for (std::size_t i = 0; i < base_t::rank(); i++) {
        if (i != 0) {
            str += ", ";
        }

        str += std::to_string(exts.extent(i));
    }

    if constexpr (base_t::rank() == 1) {
        str += ",";
    }

    return str + ")";
}

template <typename in_t>
    requires(!extents_c<in_t>)
[[nodiscard]] inline std::string to_string(in_t &&in) {
    const auto in_mds = to_const_mdspan(std::forward<in_t>(in));

    std::string str = "[";

    if constexpr (in_mds.rank() == 0) {
        if constexpr (requires { detail::value_to_string(in_mds()); }) {
            return detail::value_to_string(in_mds());

        } else {
            return "NOT_STRING_CONVERTIBLE";
        }

    } else {
        using index_t = typename decltype(in_mds)::index_type;

        for (index_t i = 0; i < in_mds.extent(0); i++) {
            if (i != 0) {
                str += ", ";
            }

            str += to_string(submdspan_from_left(in_mds, i));
        }
    }

    return str + "]";
}

} // namespace mdtensor::core
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/util/to_string.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/util/util.hpp

namespace mdtensor {

constexpr auto dyn = core::dyn;

template <typename T>
concept extents_c = core::extents_c<T>;

template <typename T>
concept mdspan_c = core::mdspan_c<T>;

template <typename T>
concept mdarray_c = core::mdarray_c<T>;

template <std::size_t start, std::size_t end>
using slice = core::slice<start, end>;

template <typename IndexType, std::size_t... Extents>
using extents = core::extents<IndexType, Extents...>;

template <typename IndexType, std::size_t Rank>
using dextents = core::dextents<IndexType, Rank>;

template <std::size_t Rank, class IndexType = std::size_t>
using dims = core::dims<Rank, IndexType>;

template <typename ElementType, typename Extents,
          typename LayoutPolicy = core::stdex::layout_right,
          typename AccessorPolicy = core::stdex::default_accessor<ElementType>>
using mdspan = core::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

constexpr auto full_extent = core::full_extent;

template <typename value_t, extents_c extent_t>
using tensor = core::tensor<value_t, extent_t>;

using Backend = core::Backend;

template <extents_c... ins_t>
[[nodiscard]] constexpr bool is_always_same_extents() noexcept {
    return core::is_always_same_extents<ins_t...>();
}

template <extents_c... ins_t>
[[nodiscard]] constexpr bool is_same_extents(ins_t &&...ins) noexcept {
    return core::is_same_extents(std::forward<ins_t>(ins)...);
}

[[nodiscard]] constexpr auto to_mdspan(auto &&io) {
    return core::to_mdspan(std::forward<decltype(io)>(io));
}

[[nodiscard]] constexpr auto to_const_mdspan(auto &&in) {
    return core::to_const_mdspan(std::forward<decltype(in)>(in));
}

[[nodiscard]] constexpr auto to_output_mdspan(auto &&out) {
    return core::to_output_mdspan(std::forward<decltype(out)>(out));
}

[[nodiscard]] constexpr auto submdspan(auto &&io, auto &&...slices) {
    return core::submdspan(std::forward<decltype(io)>(io),
                           std::forward<decltype(slices)>(slices)...);
}

template <std::size_t lspace = 0, std::size_t rspace = 0>
[[nodiscard]] constexpr auto submdspan_with_space(auto &&io, auto &&...slices) {
    return core::submdspan_with_space<lspace, rspace>(
        std::forward<decltype(io)>(io),
        std::forward<decltype(slices)>(slices)...);
}

template <std::size_t lspace = 0>
[[nodiscard]] constexpr auto submdspan_from_left(auto &&io, auto &&...slices) {
    return core::submdspan_from_left<lspace>(
        std::forward<decltype(io)>(io),
        std::forward<decltype(slices)>(slices)...);
}

template <std::size_t rspace = 0>
[[nodiscard]] constexpr auto submdspan_from_right(auto &&io, auto &&...slices) {
    return core::submdspan_from_right<rspace>(
        std::forward<decltype(io)>(io),
        std::forward<decltype(slices)>(slices)...);
}

template <extents_c exts_t>
[[nodiscard]] constexpr std::string to_string(exts_t &&exts) {
    return core::to_string(std::forward<exts_t>(exts));
}

template <typename in_t>
    requires(!extents_c<in_t>)
[[nodiscard]] constexpr std::string to_string(in_t &&in) {
    return core::to_string(std::forward<in_t>(in));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/core/core.hpp
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

template <typename dtype = double>
[[nodiscard]] constexpr auto empty(auto &&shape) {
    return core::make_tensor<dtype>(
        core::to_extents(std::forward<decltype(shape)>(shape)));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/empty.hpp

namespace mdtensor {

template <typename dtype = void, core::extents_c exts_t,
          core::arithmetic_c start_t = int, core::arithmetic_c step_t = int,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto arange(exts_t &&exts, start_t &&start = start_t{0},
                                    step_t &&step = step_t{1},
                                    out_t &&out = out_t{std::nullopt}) {
    static_assert(exts.rank() == 1, "arange only supports rank-1 extents");

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            using value_t = core::output_value_t<dtype, start_t, step_t>;

            return empty<value_t>(std::forward<decltype(exts)>(exts));

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    using value_t = typename decltype(out_md)::value_type;
    using index_t = typename decltype(out_md)::index_type;

    const value_t actual_step =
        static_cast<value_t>(start + step) - static_cast<value_t>(start);

    out_md(0) = static_cast<value_t>(start);

    for (index_t i = 1; i < out_md.extent(0); i++) {
        out_md(i) = out_md(i - 1) + static_cast<value_t>(actual_step);
    }

    return out_md;
}

template <typename dtype = void, core::arithmetic_c start_t,
          core::arithmetic_c stop_t, core::arithmetic_c step_t = int,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto arange(start_t &&start, stop_t &&stop,
                                    step_t &&step = step_t{1},
                                    out_t &&out = out_t{std::nullopt}) {
    const std::int64_t num = std::ceil((stop - start) / step);

    if (num < 0) {
        throw std::invalid_argument(
            "calculated number of elements is negative");
    }

    return arange<dtype>(core::dims<1>{static_cast<std::size_t>(num)},
                         std::forward<start_t>(start),
                         std::forward<step_t>(step), std::forward<out_t>(out));
}

template <typename dtype = void, core::arithmetic_c stop_t>
[[nodiscard]] constexpr auto arange(stop_t &&stop) {
    using start_t = typename std::remove_cvref_t<stop_t>;

    return arange<dtype>(start_t{0}, std::forward<stop_t>(stop), start_t{1});
}

template <std::size_t num, typename dtype = void,
          core::arithmetic_c start_t = int, core::arithmetic_c step_t = int,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto arange(start_t &&start = start_t{0},
                                    step_t &&step = step_t{1},
                                    out_t &&out = out_t{std::nullopt}) {
    return arange<dtype>(core::extents<std::size_t, num>{},
                         std::forward<start_t>(start),
                         std::forward<step_t>(step), std::forward<out_t>(out));
}

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


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/empty_like.hpp
/**
 * @file
 * @brief Empty-like tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

template <typename dtype = void>
[[nodiscard]] constexpr auto empty_like(auto &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using value_t = core::output_value_t<dtype, decltype(in_mds)>;

    return empty<value_t>(in_mds.extents());
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/empty_like.hpp

namespace mdtensor {
namespace ufunc {

constexpr void copy_ufunc(auto &&in, auto &&out) { out() = in(); }

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto copy(auto &&in,
                                  out_t &&out = out_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return empty_like<dtype>(in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::copy_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false>{}, in_mds, out_md);

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/copy.hpp
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
namespace ufunc {

constexpr void eye_ufunc(auto &&out, const int &k) {
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using index_t = typename decltype(out_mds)::index_type;

    for (index_t i = 0; i < out_mds.extent(0); i++) {
        for (index_t j = 0; j < out_mds.extent(1); j++) {
            out_mds(i, j) = (i + k == j) ? 1 : 0;
        }
    }
}

} // namespace ufunc

template <typename dtype = double, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto eye(auto &&shape, const int &k = 0,
                                 out_t &&out = out_t{std::nullopt}) {
    using shape_t = std::remove_cvref_t<decltype(shape)>;

    if constexpr (core::extents_c<shape_t>) {
        const auto exts =
            core::to_extents(std::forward<decltype(shape)>(shape));

        static_assert(exts.rank() >= 2, "eye requires rank >= 2");

        auto out_md = [&]() {
            if constexpr (core::nullopt_t_c<decltype(out)>) {
                return empty<dtype>(exts);

            } else {
                return core::to_output_mdspan(std::forward<decltype(out)>(out));
            }
        }();

        core::batch<backend, exts.rank() - 2>(
            [&](auto &&...elems) {
                ufunc::eye_ufunc(std::forward<decltype(elems)>(elems)..., k);
            },
            out_md);

        return out_md;

    } else if constexpr (core::integral_c<shape_t>) {
        if (shape < shape_t{0}) {
            throw std::invalid_argument("exts must be non-negative");
        }

        using index_t = std::make_unsigned_t<shape_t>;

        return eye<dtype, backend>(
            core::dextents<index_t, 2>{static_cast<index_t>(shape),
                                       static_cast<index_t>(shape)},
            k);
    }
}

template <std::size_t N, typename dtype = double,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto eye(const int &k = 0,
                                 out_t &&out = out_t{std::nullopt}) {
    return eye<dtype, backend>(core::extents<std::size_t, N, N>{}, k,
                               std::forward<out_t>(out));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/creation/eye.hpp
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
namespace ufunc {

constexpr void full_ufunc(auto &&out, auto &&val) { out() = val(); }

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto full(auto &&shape, auto &&val,
                                  out_t &&out = out_t{std::nullopt}) {
    const auto val_mds =
        core::to_const_mdspan(std::forward<decltype(val)>(val));

    using value_t = core::output_value_t<dtype, decltype(val_mds)>;

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return empty<value_t>(std::forward<decltype(shape)>(shape));

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::full_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, false, true>{}, out_md, val_mds);

    return out_md;
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

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto full_like(auto &&in, auto &&val) {
    using value_t = core::output_value_t<dtype, decltype(in)>;

    return full<value_t, backend>(in.extents(),
                                  std::forward<decltype(val)>(val));
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
namespace ufunc {

constexpr void add_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = in1() + in2();
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto add(auto &&in1, auto &&in2,
                                 out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            // check that out is not rvalue
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::add_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/add.hpp
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
namespace ufunc {

constexpr void multiply_ufunc(auto &&in1, auto &&in2, auto &&out,
                              auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = in1() * in2();
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto multiply(auto &&in1, auto &&in2,
                                      out_t &&out = out_t{std::nullopt},
                                      where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::multiply_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/multiply.hpp
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
namespace ufunc {

constexpr void subtract_ufunc(auto &&in1, auto &&in2, auto &&out,
                              auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = in1() - in2();
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto subtract(auto &&in1, auto &&in2,
                                      out_t &&out = out_t{std::nullopt},
                                      where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::subtract_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/subtract.hpp

namespace mdtensor {
namespace ufunc {

template <core::Backend backend = core::Backend::AUTO>
constexpr void linspace_ufunc(auto &&start, auto &&stop, auto &&out,
                              const bool endpoint = true) {
    const auto start_mds =
        core::to_const_mdspan(std::forward<decltype(start)>(start));
    const auto stop_mds =
        core::to_const_mdspan(std::forward<decltype(stop)>(stop));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using value_t = typename decltype(out_mds)::value_type;
    using index_t = typename decltype(out_mds)::index_type;

    const index_t num = out_mds.extent(0);

    if (num == 0) {
        return;

    } else if (num == 1) {
        if (!endpoint) {
            static_cast<void>(
                copy(start_mds, core::submdspan_from_left(out_mds)));

        } else {
            static_cast<void>(
                copy(stop_mds, core::submdspan_from_left(out_mds)));
        }

    } else {
        const value_t scale = value_t{1} / (endpoint ? num - 1 : num);

        const auto step = multiply<value_t, backend>(
            subtract<value_t, backend>(stop_mds, start_mds), scale);

        static_cast<void>(copy(start_mds, out_mds));

        for (index_t i = 1; i < num; i++) {
            static_cast<void>(add<void, backend>(
                core::submdspan_from_left(out_mds, i),
                multiply<value_t, backend>(step, static_cast<value_t>(i)),
                core::submdspan_from_left(out_mds, i)));
        }

        if (endpoint) {
            // Ensure that the last element is exactly equal to the stop value
            static_cast<void>(
                copy(stop_mds, core::submdspan_from_left(out_mds, num - 1)));
        }
    }
}

} // namespace ufunc

template <std::int64_t axis = 0, typename dtype = void,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto linspace(auto &&shape, auto &&start, auto &&stop,
                                      const bool endpoint = true,
                                      out_t &&out = out_t{std::nullopt}) {
    const auto exts = core::to_extents(std::forward<decltype(shape)>(shape));

    static_assert(exts.rank() == 1,
                  "The extents for linspace must be a 1D tensor.");

    const auto [bcasts, bexts] = core::broadcast(
        std::index_sequence<0, 0>{}, std::integer_sequence<bool, true, true>{},
        std::forward<decltype(start)>(start),
        std::forward<decltype(stop)>(stop));
    const auto start_bcast = std::get<0>(bcasts);
    const auto stop_bcast = std::get<1>(bcasts);

    constexpr std::size_t baxis =
        static_cast<std::size_t>(core::bounding_index(axis, bexts.rank()));
    constexpr std::size_t out_urank = bexts.rank() + 1 - baxis;

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            using value_t =
                core::output_value_t<dtype,
                                     typename decltype(start_bcast)::value_type,
                                     typename decltype(stop_bcast)::value_type>;

            return empty<value_t>(core::compose_extents(
                core::slice_extents_from_left<baxis>(bexts), exts,
                core::slice_extents_from_right<out_urank - 1>(bexts)));

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [&](auto &&...elems) {
            ufunc::linspace_ufunc<core::Backend::NATIVE>(
                std::forward<decltype(elems)>(elems)..., endpoint);
        },
        std::index_sequence<out_urank - 1, out_urank - 1, out_urank>{},
        std::integer_sequence<bool, true, true, false>{}, start_bcast,
        stop_bcast, out_md);

    return out_md;
}

template <std::size_t num, std::int64_t axis = 0, typename dtype = void,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto linspace(auto &&start, auto &&stop,
                                      const bool endpoint = true,
                                      out_t &&out = out_t{std::nullopt}) {
    return linspace<axis, dtype, backend>(
        core::extents<std::size_t, num>{}, std::forward<decltype(start)>(start),
        std::forward<decltype(stop)>(stop), endpoint,
        std::forward<decltype(out)>(out));
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

template <typename dtype = double, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto ones(auto &&shape,
                                  out_t &&out = out_t{std::nullopt}) {
    return full<dtype, backend>(std::forward<decltype(shape)>(shape), 1,
                                std::forward<decltype(out)>(out));
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

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto ones_like(auto &&in) {
    return full_like<dtype, backend>(std::forward<decltype(in)>(in), 1);
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

template <typename dtype = double, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto zeros(auto &&shape,
                                   out_t &&out = out_t{std::nullopt}) {
    return full<dtype, backend>(std::forward<decltype(shape)>(shape), 0,
                                std::forward<decltype(out)>(out));
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

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto zeros_like(auto &&in) {
    return full_like<dtype, backend>(std::forward<decltype(in)>(in), 0);
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


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/cholesky.hpp
/**
 * @file
 * @brief Cholesky decomposition utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


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
namespace ufunc {

template <std::floating_point dtype>
[[nodiscard]] constexpr dtype
sqrt_newton_raphson(const dtype &x, const dtype &curr, const dtype &prev) {
    return (curr == prev)
               ? curr
               : sqrt_newton_raphson(x, (curr + x / curr) / (dtype)2, curr);
}

constexpr void sqrt_ufunc_native(auto &&in, auto &&out) {
    using calc_t = core::common_data_type_t<decltype(in()), float>;

    if constexpr (requires {
                      { std::isnan(in()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in())) {
            out() = in();
            return;
        }
    }

    if constexpr (requires {
                      { std::isinf(in()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isinf(in())) {
            out() = in();
            return;
        }
    }

    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(in())>, bool>) {
        out() = static_cast<calc_t>(in());
        return;

    } else {
        out() = (in() >= 0 && in() < std::numeric_limits<calc_t>::infinity())
                    ? sqrt_newton_raphson(static_cast<calc_t>(in()),
                                          static_cast<calc_t>(in()),
                                          static_cast<calc_t>(0))
                    : std::numeric_limits<calc_t>::quiet_NaN();
    }
}

constexpr void sqrt_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

#ifdef REAL_GCC
    if (!std::is_constant_evaluated()) {
        if constexpr (requires { out() = std::sqrt(in()); }) {
            out() = std::sqrt(in());
            return;
        }
    }

#endif

    sqrt_ufunc_native(std::forward<decltype(in)>(in),
                      std::forward<decltype(out)>(out));
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto sqrt(auto &&in, out_t &&out = out_t{std::nullopt},
                                  where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return empty_like<dtype>(in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::sqrt_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/sqrt.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/util/fill.hpp
/**
 * @file
 * @brief Fill utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace ufunc {

constexpr void fill_ufunc(auto &&out, auto &&val) { out() = val; }

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void fill(auto &&out, auto &&val) {
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    core::batch<backend, out_mds.rank()>(
        [&](auto &&...elems) {
            ufunc::fill_ufunc(std::forward<decltype(elems)>(elems)...,
                              std::forward<decltype(val)>(val));
        },
        out_mds);
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/util/fill.hpp

namespace mdtensor::linalg {
namespace ufunc {

[[nodiscard]] constexpr bool cholesky_upper_ufunc(auto &&in, auto &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using index_t = typename decltype(in_mds)::index_type;
    using value_t = typename decltype(out_mds)::value_type;

    const index_t n = in_mds.extent(0);

    // create zero matrix for output
    fill(out_mds, 0);

    // perform the upper Cholesky decomposition
    for (index_t i = 0; i < n; i++) {
        for (index_t j = i; j < n; j++) {
            value_t sum = 0;

            for (index_t k = 0; k < i; k++) {
                sum += out_mds(k, i) * out_mds(k, j);
            }

            if (i == j) {
                const auto diag = in_mds(i, i) - sum;

                if (diag <= 0) {
                    return false;
                }

                out_mds(i, i) = sqrt(diag);

            } else {
                if (out_mds(i, i) <= 0) {
                    return false;
                }

                out_mds(i, j) = (in_mds(i, j) - sum) / out_mds(i, i);
            }
        }
    }

    return true;
}

[[nodiscard]] constexpr bool cholesky_lower_ufunc(auto &&in, auto &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using index_t = typename decltype(in_mds)::index_type;
    using value_t = typename decltype(out_mds)::value_type;

    const index_t n = in_mds.extent(0);

    // create zero matrix for output
    fill(out_mds, 0);

    // perform the cholesky decomposition
    for (index_t i = 0; i < n; i++) {
        for (index_t j = 0; j <= i; j++) {
            value_t sum = 0;

            for (index_t k = 0; k < j; k++) {
                sum += out_mds(i, k) * out_mds(j, k);
            }

            if (i == j) {
                const auto diag = in_mds(i, i) - sum;

                if (diag <= 0) {
                    return false;
                }

                out_mds(i, j) = sqrt(diag);

            } else {
                if (out_mds(j, j) <= 0) {
                    return false;
                }

                out_mds(i, j) = (in_mds(i, j) - sum) / out_mds(j, j);
            }
        }
    }

    return true;
}

template <bool upper>
[[nodiscard]] constexpr bool cholesky_ufunc(auto &&in, auto &&out) {
    if constexpr (upper) {
        return cholesky_upper_ufunc(std::forward<decltype(in)>(in),
                                    std::forward<decltype(out)>(out));

    } else {
        return cholesky_lower_ufunc(std::forward<decltype(in)>(in),
                                    std::forward<decltype(out)>(out));
    }
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void cholesky_to(auto &&in, auto &&out, auto &&valid,
                           const bool upper = false) {
    const auto run_batch = [&]<bool upper_v>() {
        core::batch_with_broadcast<backend>(
            [](auto &&in, auto &&out, auto &&valid) {
                valid() = ufunc::cholesky_ufunc<upper_v>(
                    std::forward<decltype(in)>(in),
                    std::forward<decltype(out)>(out));
            },
            std::index_sequence<2, 2, 0>{},
            std::integer_sequence<bool, true, false, false>{},
            std::forward<decltype(in)>(in), std::forward<decltype(out)>(out),
            std::forward<decltype(valid)>(valid));
    };

    if (upper) {
        run_batch.template operator()<true>();

    } else {
        run_batch.template operator()<false>();
    }
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto cholesky(auto &&in, const bool upper = false) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out = empty_like(in_mds);
    auto valid = core::make_broadcasted_tensor<bool>(
        std::index_sequence<2>{}, core::extents<std::uint8_t>{}, in_mds);

    cholesky_to<backend>(in_mds, out, valid, upper);

    return std::pair{out, valid};
}

} // namespace mdtensor::linalg
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/cholesky.hpp
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
namespace ufunc {

constexpr void absolute_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    if constexpr (std::is_signed_v<std::remove_cvref_t<decltype(in())>>) {
#ifdef REAL_GCC // NOTE: std::abs is not constexpr in clang 16.
        out() = std::abs(in());
#else
        out() = in() < 0 ? -in() : in();
#endif

    } else {
        out() = in();
    }
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto absolute(auto &&in,
                                      out_t &&out = out_t{std::nullopt},
                                      where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return empty_like<dtype>(in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::absolute_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/absolute.hpp

#ifdef MDTENSOR_USE_EIGEN
#include "../core/eigen/eigen.hpp"
#endif

namespace mdtensor::linalg {
namespace ufunc {

[[nodiscard]] constexpr bool inv_native(auto &&in, auto &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using index_t = typename decltype(in_mds)::index_type;

    const index_t n = in_mds.extent(0);

    auto in_copy = copy(in_mds);
    static_cast<void>(eye(out_mds.extents(), 0, out_mds));

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

[[nodiscard]] constexpr bool inv_ufunc(auto &&in, auto &&out) {
#ifdef MDTENSOR_USE_EIGEN
#if __cplusplus >= 202302L // TODO: Impliement for C++20
    if constexpr (core::eigen::eigen_mappable_c<in_t> &&
                  core::eigen::eigen_mappable_c<out_t>) {
        if (!std::is_constant_evaluated()) {
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

    return inv_native(in, out);
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void inv_to(auto &&in, auto &&out, auto &&valid) {
    core::batch_with_broadcast<backend>(
        [](auto &&in, auto &&out, auto &&valid) {
            valid() = ufunc::inv_ufunc(std::forward<decltype(in)>(in),
                                       std::forward<decltype(out)>(out));
        },
        std::index_sequence<2, 2, 0>{},
        std::integer_sequence<bool, true, false, false>{},
        std::forward<decltype(in)>(in), std::forward<decltype(out)>(out),
        std::forward<decltype(valid)>(valid));
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto inv(auto &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out = empty_like<dtype>(in_mds);
    auto valid = core::make_broadcasted_tensor<bool>(
        std::index_sequence<2>{}, core::extents<std::uint8_t>{}, in_mds);

    inv_to<backend>(in_mds, out, valid);

    return std::pair{out, valid};
}

} // namespace mdtensor::linalg
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/inv.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/lu.hpp
/**
 * @file
 * @brief LU decomposition utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor::linalg {
namespace ufunc {

constexpr void lu_p_indices_ufunc(auto &&in, auto &&p_indices, auto &&l,
                                  auto &&u) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto p_indices_mds =
        core::to_output_mdspan(std::forward<decltype(p_indices)>(p_indices));
    const auto l_mds = core::to_output_mdspan(std::forward<decltype(l)>(l));
    const auto u_mds = core::to_output_mdspan(std::forward<decltype(u)>(u));

    using in_mds_t = decltype(in_mds);
    using p_indices_mds_t = decltype(p_indices_mds);

    using index_t = typename in_mds_t::index_type;

    constexpr std::size_t m_s = in_mds_t::static_extent(0);

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
    auto row_order = empty<index_t>(core::extents<index_t, m_s>{m});
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

constexpr void lu_full_ufunc(auto &&in, auto &&p, auto &&l, auto &&u) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto p_mds = core::to_output_mdspan(std::forward<decltype(p)>(p));
    const auto l_mds = core::to_output_mdspan(std::forward<decltype(l)>(l));
    const auto u_mds = core::to_output_mdspan(std::forward<decltype(u)>(u));

    using in_mds_t = decltype(in_mds);
    using p_mds_t = decltype(p_mds);

    using index_t = typename p_mds_t::index_type;

    constexpr std::size_t m_s = in_mds_t::static_extent(0);

    const index_t m = in_mds.extent(0);

    auto p_indices = empty<index_t>(core::extents<index_t, m_s>{m});

    lu_p_indices_ufunc(in_mds, p_indices, l_mds, u_mds);

    for (index_t i = 0; i < m; i++) {
        for (index_t j = 0; j < m; j++) {
            p_mds(i, j) = (j == p_indices(i) ? 1 : 0);
        }
    }
}

constexpr void lu_permute_l_ufunc(auto &&in, auto &&pl, auto &&u) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto pl_mds = core::to_output_mdspan(std::forward<decltype(pl)>(pl));
    const auto u_mds = core::to_output_mdspan(std::forward<decltype(u)>(u));

    using in_mds_t = decltype(in_mds);
    using pl_mds_t = decltype(pl_mds);

    using index_t = typename in_mds_t::index_type;
    using value_t = typename pl_mds_t::value_type;

    constexpr std::size_t m_s = in_mds_t::static_extent(0);
    constexpr std::size_t n_s = in_mds_t::static_extent(1);
    constexpr std::size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(0);
    const index_t n = in_mds.extent(1);
    const index_t k = m < n ? m : n;

    auto p_indices = empty<index_t>(core::extents<index_t, m_s>{m});
    auto l = empty<value_t>(core::extents<index_t, m_s, k_s>{m, k});

    lu_p_indices_ufunc(in_mds, p_indices, l, u_mds);

    // Apply the permutation to L
    for (index_t i = 0; i < m; i++) {
        for (index_t j = 0; j < k; j++) {
            pl_mds(i, j) = l(p_indices(i), j);
        }
    }
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void lu_p_indices_to(auto &&in, auto &&p_indices, auto &&l,
                               auto &&u) {
    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::lu_p_indices_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 1, 2, 2>{},
        std::integer_sequence<bool, true, false, false, false>{},
        std::forward<decltype(in)>(in),
        std::forward<decltype(p_indices)>(p_indices),
        std::forward<decltype(l)>(l), std::forward<decltype(u)>(u));
}

template <core::Backend backend = core::Backend::AUTO>
constexpr void lu_full_to(auto &&in, auto &&p, auto &&l, auto &&u) {
    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::lu_full_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 2, 2, 2>{},
        std::integer_sequence<bool, true, false, false, false>{},
        std::forward<decltype(in)>(in), std::forward<decltype(p)>(p),
        std::forward<decltype(l)>(l), std::forward<decltype(u)>(u));
}

template <core::Backend backend = core::Backend::AUTO>
constexpr void lu_permute_l_to(auto &&in, auto &&pl, auto &&u) {
    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::lu_permute_l_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 2, 2>{},
        std::integer_sequence<bool, true, false, false>{},
        std::forward<decltype(in)>(in), std::forward<decltype(pl)>(pl),
        std::forward<decltype(u)>(u));
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto lu_p_indices(auto &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using in_mds_t = decltype(in_mds);
    using index_t = typename in_mds_t::index_type;

    constexpr std::size_t rank = in_mds_t::rank();

    constexpr std::size_t m_s = in_mds_t::static_extent(rank - 2);
    constexpr std::size_t n_s = in_mds_t::static_extent(rank - 1);
    constexpr std::size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(rank - 2);
    const index_t n = in_mds.extent(rank - 1);
    const index_t k = m < n ? m : n;

    auto outs = core::make_broadcasted_tensors<dtype>(
        std::index_sequence<2>{},
        std::tuple{extents<index_t, m_s>{m},
                   core::extents<index_t, m_s, k_s>{m, k},
                   core::extents<index_t, k_s, n_s>{k, n}},
        in_mds);

    lu_p_indices_to<backend>(in_mds, std::get<0>(outs), std::get<1>(outs),
                             std::get<2>(outs));

    return outs;
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto lu_full(auto &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using in_mds_t = decltype(in_mds);
    using index_t = typename in_mds_t::index_type;

    constexpr std::size_t rank = in_mds_t::rank();

    constexpr std::size_t m_s = in_mds_t::static_extent(rank - 2);
    constexpr std::size_t n_s = in_mds_t::static_extent(rank - 1);
    constexpr std::size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(rank - 2);
    const index_t n = in_mds.extent(rank - 1);
    const index_t k = m < n ? m : n;

    auto outs = core::make_broadcasted_tensors<dtype>(
        std::index_sequence<2>{},
        std::tuple{extents<index_t, m_s, m_s>{m, m},
                   core::extents<index_t, m_s, k_s>{m, k},
                   core::extents<index_t, k_s, n_s>{k, n}},
        in_mds);

    lu_full_to<backend>(in_mds, std::get<0>(outs), std::get<1>(outs),
                        std::get<2>(outs));

    return outs;
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto lu_permute_l(auto &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using in_mds_t = decltype(in_mds);
    using index_t = typename in_mds_t::index_type;

    constexpr std::size_t rank = in_mds_t::rank();

    constexpr std::size_t m_s = in_mds_t::static_extent(rank - 2);
    constexpr std::size_t n_s = in_mds_t::static_extent(rank - 1);
    constexpr std::size_t k_s = [] {
        if constexpr (m_s == dyn || n_s == dyn) {
            return dyn;

        } else {
            return m_s < n_s ? m_s : n_s;
        }
    }();

    const index_t m = in_mds.extent(rank - 2);
    const index_t n = in_mds.extent(rank - 1);
    const index_t k = m < n ? m : n;

    auto outs = core::make_broadcasted_tensors<dtype>(
        std::index_sequence<2>{},
        std::tuple{extents<index_t, m_s, k_s>{m, k},
                   core::extents<index_t, k_s, n_s>{k, n}},
        in_mds);

    lu_permute_l_to<backend>(in_mds, std::get<0>(outs), std::get<1>(outs));

    return outs;
}

template <bool permute_l = false, bool p_indices = false, typename dtype = void,
          core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto lu(auto &&in) {
    static_assert(!(permute_l && p_indices),
                  "lu cannot return both permuted L and P indices.");

    if constexpr (permute_l) {
        return lu_permute_l<dtype, backend>(std::forward<decltype(in)>(in));

    } else if constexpr (p_indices) {
        return lu_p_indices<dtype, backend>(std::forward<decltype(in)>(in));

    } else {
        return lu_full<dtype, backend>(std::forward<decltype(in)>(in));
    }
}

} // namespace mdtensor::linalg
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



//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/expand_dims.hpp
/**
 * @file
 * @brief Dimension expansion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

template <std::integral axes_t, axes_t... axes>
[[nodiscard]] constexpr auto
expand_dims(auto &&in, std::integer_sequence<axes_t, axes...>) {
    return core::expand_dims(std::forward<decltype(in)>(in),
                             std::integer_sequence<axes_t, axes...>{});
}

template <std::int64_t... axes>
[[nodiscard]] constexpr auto expand_dims(auto &&in) {
    return core::expand_dims<axes...>(std::forward<decltype(in)>(in));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/expand_dims.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/matvec.hpp
/**
 * @file
 * @brief Matrix-vector multiplication utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



#ifdef MDTENSOR_USE_EIGEN
#include "../core/eigen/eigen.hpp"
#endif

namespace mdtensor::linalg {
namespace ufunc {

constexpr void matvec_native_noalias(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using out_index_t = typename decltype(out_mds)::index_type;
    using in1_index_t = typename decltype(in1_mds)::index_type;

    for (out_index_t i = 0; i < out_mds.extent(0); i++) {
        out_mds(i) = 0;

        for (in1_index_t j = 0; j < in1_mds.extent(1); j++) {
            out_mds(i) += in1_mds(i, j) * in2_mds(j);
        }
    }
}

constexpr void matvec_naive(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    if (std::is_constant_evaluated()) {
        auto out_tmp = empty_like(out_mds);
        matvec_native_noalias(in1_mds, in2_mds, out_tmp);
        static_cast<void>(copy(out_tmp, out_mds));
        return;
    }

    bool need_copy = false;

    if constexpr (requires {
                      in1_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in1_mds.data_handle() == out_mds.data_handle()) {
            need_copy = true;
        }
    }

    if constexpr (requires {
                      in2_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in2_mds.data_handle() == out_mds.data_handle()) {
            need_copy = true;
        }
    }

    if (!need_copy) {
        matvec_native_noalias(in1_mds, in2_mds, out_mds);

    } else {
        auto out_tmp = empty_like(out_mds);
        matvec_native_noalias(in1_mds, in2_mds, out_tmp);
        static_cast<void>(copy(out_tmp, out_mds));
    }
}

constexpr void matvec_ufunc(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

#ifdef MDTENSOR_USE_EIGEN
#if __cplusplus >= 202302L // TODO: Impliement for C++20
    using in1_mds_t = decltype(in1_mds);
    using in2_mds_t = decltype(in2_mds);
    using out_mds_t = decltype(out_mds);

    if constexpr (core::eigen::eigen_mappable_c<in1_mds_t> &&
                  core::eigen::eigen_mappable_c<in2_mds_t> &&
                  core::eigen::eigen_mappable_c<out_mds_t>) {
        if (!std::is_constant_evaluated() &&
            8 <= out_mds.extent(0) + out_mds.extent(1)) {
            using value_t = core::common_data_type_t<
                typename std::remove_cvref_t<in1_mds_t>::value_type,
                typename std::remove_cvref_t<in2_mds_t>::value_type>;

            const auto ein1 =
                core::eigen::to_eigen(in1_mds).template cast<value_t>();
            const auto ein2 =
                core::eigen::to_eigen(in2_mds).template cast<value_t>();
            auto eout = core::eigen::to_eigen(out_mds);

            eout =
                (ein1 * ein2)
                    .template cast<
                        typename std::remove_cvref_t<out_mds_t>::value_type>();

            return;
        }
    }

#else
    assert(false && "Eigen inverse not implemented for C++20");

#endif
#endif

    matvec_naive(in1_mds, in2_mds, out_mds);
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void matvec_to(auto &&in1, auto &&in2, auto &&out) {
    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::matvec_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2, 1, 1>{},
        std::integer_sequence<bool, true, true, false>{},
        std::forward<decltype(in1)>(in1), std::forward<decltype(in2)>(in2),
        std::forward<decltype(out)>(out));
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto matvec(auto &&in1, auto &&in2) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    const auto uin1_exts = core::slice_extents_from_right<2>(in1_mds.extents());
    const auto uin2_exts = core::slice_extents_from_right<2>(in2_mds.extents());
    const auto uout_exts = core::extents<
        core::common_index_type_t<typename decltype(uin1_exts)::index_type,
                                  typename decltype(uin2_exts)::index_type>,
        decltype(uin1_exts)::static_extent(0)>{uin1_exts.extent(0)};

    auto out = core::make_broadcasted_tensor<dtype>(
        std::index_sequence<2, 1>{}, uout_exts, in1_mds, in2_mds);

    matvec_to<backend>(in1_mds, in2_mds, out);

    return out;
}

} // namespace mdtensor::linalg
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/matvec.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/vecmat.hpp
/**
 * @file
 * @brief Vector-matrix multiplication utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



#ifdef MDTENSOR_USE_EIGEN
#include "../core/eigen/eigen.hpp"
#endif

namespace mdtensor::linalg {
namespace ufunc {

constexpr void vecmat_naive_noalias(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using out_index_t = typename decltype(out_mds)::index_type;
    using in1_index_t = typename decltype(in1_mds)::index_type;

    for (out_index_t i = 0; i < out_mds.extent(0); i++) {
        out_mds(i) = 0;

        for (in1_index_t j = 0; j < in1_mds.extent(0); j++) {
            out_mds(i) += in1_mds(j) * in2_mds(j, i);
        }
    }
}

constexpr void vecmat_naive(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    if (std::is_constant_evaluated()) {
        auto out_tmp = empty_like(out_mds);
        vecmat_naive_noalias(in1_mds, in2_mds, out_tmp);
        static_cast<void>(copy(out_tmp, out_mds));
        return;
    }

    bool need_copy = false;

    if constexpr (requires {
                      in1_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in1_mds.data_handle() == out_mds.data_handle()) {
            need_copy = true;
        }
    }

    if constexpr (requires {
                      in2_mds.data_handle() == out_mds.data_handle();
                  }) {
        if (in2_mds.data_handle() == out_mds.data_handle()) {
            need_copy = true;
        }
    }

    if (!need_copy) {
        vecmat_naive_noalias(in1_mds, in2_mds, out_mds);

    } else {
        auto out_tmp = empty_like(out_mds);
        vecmat_naive_noalias(in1_mds, in2_mds, out_tmp);
        static_cast<void>(copy(out_tmp, out_mds));
    }
}

constexpr void vecmat_ufunc(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

#ifdef MDTENSOR_USE_EIGEN
#if __cplusplus >= 202302L // TODO: Impliement for C++20
    using in1_mds_t = decltype(in1_mds);
    using in2_mds_t = decltype(in2_mds);
    using out_mds_t = decltype(out_mds);

    if constexpr (core::eigen::eigen_mappable_c<in1_mds_t> &&
                  core::eigen::eigen_mappable_c<in2_mds_t> &&
                  core::eigen::eigen_mappable_c<out_mds_t>) {
        if (!std::is_constant_evaluated() &&
            8 <= out_mds.extent(0) + out_mds.extent(1)) {
            using value_t = core::common_data_type_t<
                typename std::remove_cvref_t<in1_mds_t>::value_type,
                typename std::remove_cvref_t<in2_mds_t>::value_type>;

            const auto ein1 =
                core::eigen::to_eigen(in1_mds).template cast<value_t>();
            const auto ein2 =
                core::eigen::to_eigen(in2_mds).template cast<value_t>();
            auto eout = core::eigen::to_eigen(out_mds);

            eout =
                (ein1 * ein2)
                    .template cast<
                        typename std::remove_cvref_t<out_mds_t>::value_type>();

            return;
        }
    }

#else
    assert(false && "Eigen inverse not implemented for C++20");

#endif
#endif

    vecmat_naive(in1_mds, in2_mds, out_mds);
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void vecmat_to(auto &&in1, auto &&in2, auto &&out) {
    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::vecmat_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<1, 2, 1>{},
        std::integer_sequence<bool, true, true, false>{},
        std::forward<decltype(in1)>(in1), std::forward<decltype(in2)>(in2),
        std::forward<decltype(out)>(out));
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto vecmat(auto &&in1, auto &&in2) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    const auto uin1_exts = core::slice_extents_from_right<2>(in1_mds.extents());
    const auto uin2_exts = core::slice_extents_from_right<2>(in2_mds.extents());
    const auto uout_exts = core::extents<
        core::common_data_type_t<typename decltype(uin1_exts)::index_type,
                                 typename decltype(uin2_exts)::index_type>,
        decltype(uin2_exts)::static_extent(1)>{uin2_exts.extent(1)};

    auto out = core::make_broadcasted_tensor<dtype>(
        std::index_sequence<1, 2>{}, uout_exts, in1_mds, in2_mds);

    vecmat_to<backend>(in1_mds, in2_mds, out);

    return out;
}

} // namespace mdtensor::linalg
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/vecmat.hpp

#ifdef MDTENSOR_USE_EIGEN
#include "../core/eigen/eigen.hpp"
#endif

// TODO: modifiy

namespace mdtensor::linalg {
namespace ufunc {

constexpr void matmul_ufunc_native_noalias(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using out_index_t = typename decltype(out_mds)::index_type;
    using in1_index_t = typename decltype(in1_mds)::index_type;

    for (out_index_t i = 0; i < out_mds.extent(0); i++) {
        for (out_index_t j = 0; j < out_mds.extent(1); j++) {
            out_mds(i, j) = 0;

            for (in1_index_t k = 0; k < in1_mds.extent(1); k++) {
                out_mds(i, j) += in1_mds(i, k) * in2_mds(k, j);
            }
        }
    }
}

constexpr void matmul_ufunc_native(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    bool need_copy = false;

    if (std::is_constant_evaluated()) {
        need_copy = true;

    } else if ((void *)in1_mds.data_handle() == (void *)out_mds.data_handle() ||
               (void *)in2_mds.data_handle() == (void *)out_mds.data_handle()) {
        need_copy = true;
    }

    if (!need_copy) {
        matmul_ufunc_native_noalias(in1_mds, in2_mds, out_mds);

    } else {
        auto out_tmp = empty_like(out_mds);
        matmul_ufunc_native_noalias(in1_mds, in2_mds, out_tmp);
        static_cast<void>(copy(out_tmp, out_mds));
    }
}

#ifdef MDTENSOR_USE_EIGEN

template <core::mdspan_c in1_t, core::mdspan_c in2_t, core::mdspan_c out_t>
    requires(core::eigen::eigen_mappable_c<in1_t> &&
             core::eigen::eigen_mappable_c<in2_t> &&
             core::eigen::eigen_mappable_c<out_t>)
inline void matmul_ufunc_eigen(const in1_t &in1, const in2_t &in2,
                               const out_t &out) {
    using value_t = core::common_data_type_t<typename in1_t::value_type,
                                             typename in2_t::value_type>;

    const auto ein1 = core::eigen::to_eigen(in1);
    const auto ein2 = core::eigen::to_eigen(in2);
    auto eout = core::eigen::to_eigen(out);

    eout = (ein1.template cast<value_t>() * ein2.template cast<value_t>())
               .template cast<typename out_t::value_type>();
}

#endif

constexpr core::Backend matmul_auto_backend(auto &&in1, auto &&in2,
                                            auto &&out) {
    if (std::is_constant_evaluated()) {
        return core::Backend::NATIVE;
    }

#ifdef MDTENSOR_USE_EIGEN
    if constexpr (core::eigen::eigen_mappable_c<decltype(in1)> &&
                  core::eigen::eigen_mappable_c<decltype(in2)> &&
                  core::eigen::eigen_mappable_c<decltype(out)>) {
        return core::Backend::EIGEN;
    }
#endif

    return core::Backend::NATIVE;
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void matmul_to(auto &&in1, auto &&in2, auto &&out) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    constexpr bool is_in1_mds_1d = (in1_mds.rank() == 1);
    constexpr bool is_in2_mds_1d = (in2_mds.rank() == 1);

    if constexpr (is_in1_mds_1d && !is_in2_mds_1d) {
        vecmat_to<backend>(in1_mds, in2_mds, out_mds);

    } else if constexpr (!is_in1_mds_1d && is_in2_mds_1d) {
        matvec_to<backend>(in1_mds, in2_mds, out_mds);

    } else {
        const auto be = backend;
        // constexpr auto be =
        //     (backend == core::Backend::AUTO)
        //         ?
        // ufunc::matmul_auto_backend(std::forward<decltype(in1)>(in1),
        // std::forward<decltype(in2)>(in2),
        // std::forward<decltype(out)>(out))
        //         : backend;

        if (
#ifdef MDTENSOR_USE_EIGEN
            be == core::Backend::EIGEN
#else
            false
#endif
        ) {
#ifdef MDTENSOR_USE_EIGEN
            core::batch_with_broadcast<core::Backend::NATIVE>(
                [](auto &&...elems) {
                    ufunc::matmul_ufunc_eigen(
                        std::forward<decltype(elems)>(elems)...);
                },
                std::index_sequence<2, 2, 2>{},
                std::integer_sequence<bool, true, true, false>{},
                std::forward<decltype(in1)>(in1),
                std::forward<decltype(in2)>(in2),
                std::forward<decltype(out)>(out));
#endif

        } else if (be == core::Backend::NATIVE) {
            core::batch_with_broadcast<core::Backend::NATIVE>(
                [](auto &&...elems) {
                    ufunc::matmul_ufunc_native(
                        std::forward<decltype(elems)>(elems)...);
                },
                std::index_sequence<2, 2, 2>{},
                std::integer_sequence<bool, true, true, false>{},
                std::forward<decltype(in1)>(in1),
                std::forward<decltype(in2)>(in2),
                std::forward<decltype(out)>(out));

        } else {
            core::batch_with_broadcast<core::Backend::NATIVE>(
                [](auto &&...elems) {
                    ufunc::matmul_ufunc_native(
                        std::forward<decltype(elems)>(elems)...);
                },
                std::index_sequence<2, 2, 2>{},
                std::integer_sequence<bool, true, true, false>{},
                std::forward<decltype(in1)>(in1),
                std::forward<decltype(in2)>(in2),
                std::forward<decltype(out)>(out));
        }
    }
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto matmul(auto &&in1, auto &&in2) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    constexpr bool is_in1_mds_1d = (in1_mds.rank() == 1);
    constexpr bool is_in2_mds_1d = (in2_mds.rank() == 1);

    if constexpr (is_in1_mds_1d && !is_in2_mds_1d) {
        return vecmat<dtype, backend>(in1_mds, in2_mds);

    } else if constexpr (!is_in1_mds_1d && is_in2_mds_1d) {
        return matvec<dtype, backend>(in1_mds, in2_mds);

    } else {
        const auto uin1_exts =
            core::slice_extents_from_right<2>(in1_mds.extents());
        const auto uin2_exts =
            core::slice_extents_from_right<2>(in2_mds.extents());
        const auto uout_exts =
            core::compose_extents(core::slice_extents_from_left<1>(uin1_exts),
                                  core::slice_extents_from_right<1>(uin2_exts));

        auto out = core::make_broadcasted_tensor<dtype>(
            std::index_sequence<uin1_exts.rank(), uin2_exts.rank()>{},
            uout_exts, in1_mds, in2_mds);

        matmul_to<backend>(in1_mds, in2_mds, out);

        return out;
    }
}

} // namespace mdtensor::linalg
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/matmul.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/norm.hpp
/**
 * @file
 * @brief Vector norm (L2) utilities for mdtensor (linalg).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/sum.hpp
/**
 * @file
 * @brief Sum reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO, std::integral axes_t,
          axes_t... axes, typename out_t = std::nullopt_t,
          typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto sum(auto &&in,
                                 std::integer_sequence<axes_t, axes...>,
                                 out_t &&out = out_t{std::nullopt},
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_reduced_tensor<dtype, keepdims>(
                std::integer_sequence<axes_t, axes...>{},
                std::index_sequence<0>{}, core::extents<std::uint8_t>{},
                in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    // TODO: move to reduce
    if constexpr (core::nullopt_t_c<decltype(initial)>) {
        fill<backend>(out_md, 0);

    } else {
        fill<backend>(out_md, std::forward<decltype(initial)>(initial));
    }

    core::reduce<keepdims>(
        [](auto &&in, auto &&out, auto &&where) {
            static_cast<void>(
                add<void, backend>(std::forward<decltype(in)>(in),
                                   std::forward<decltype(out)>(out),
                                   std::forward<decltype(out)>(out),
                                   std::forward<decltype(where)>(where)));
        },
        std::integer_sequence<axes_t, axes...>{},
        std::index_sequence<0, 0, 0>{},
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

template <std::int64_t axis, typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto sum(auto &&in, out_t &&out = out_t{std::nullopt},
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return sum<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in),
        std::integer_sequence<std::int64_t, axis>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial),
        std::forward<decltype(where)>(where));
}

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto sum(auto &&in, out_t &&out = out_t{std::nullopt},
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return sum<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in), std::index_sequence<>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial),
        std::forward<decltype(where)>(where));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/sum.hpp

namespace mdtensor::linalg {
namespace ufunc {

constexpr void norm_ufunc(auto &&in, auto &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using index_t = typename decltype(in_mds)::index_type;

    out_mds() = 0;
    for (index_t i = 0; i < in_mds.extent(0); i++) {
        out_mds() += in_mds(i) * in_mds(i);
    }

    if (out_mds() > 0) {
        out_mds() = sqrt(out_mds());
    }
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void norm_to(auto &&in, auto &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    if constexpr (backend == core::Backend::SIMD) {
        static_cast<void>(sum<-1, void, false, backend>(
            multiply<void, backend>(in_mds, in_mds), out_mds));
        static_cast<void>(sqrt<void, backend>(out_mds, out_mds));

    } else {
        core::batch_with_broadcast<backend>(
            [](auto &&...elems) {
                ufunc::norm_ufunc(std::forward<decltype(elems)>(elems)...);
            },
            std::index_sequence<1, 0>{},
            std::integer_sequence<bool, true, false>{}, in_mds, out_mds);
    }
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto norm(auto &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out = core::make_broadcasted_tensor<dtype>(
        std::index_sequence<1>{}, core::extents<std::uint8_t>{}, in_mds);

    norm_to<backend>(in_mds, out);

    return out;
}

} // namespace mdtensor::linalg
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



namespace mdtensor::linalg {
namespace ufunc {

[[nodiscard]] constexpr bool solve_ufunc(auto &&a, auto &&b, auto &&x) {
    const auto a_mds = core::to_const_mdspan(std::forward<decltype(a)>(a));
    const auto b_mds = core::to_const_mdspan(std::forward<decltype(b)>(b));
    const auto x_mds = core::to_output_mdspan(std::forward<decltype(x)>(x));

    using index_t = typename decltype(a_mds)::index_type;

    const index_t n = a_mds.extent(0);

    // LU decomposition of A
    const auto [p_indices, l, u] = lu_p_indices(a_mds);

    // check singularity
    for (index_t idx = 0; idx < n; idx++) {
        if (u(idx, idx) == 0) {
            return false;
        }
    }

    if constexpr (b_mds.rank() == 1) {
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

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void solve_to(auto &&a, auto &&b, auto &&x, auto &&valid) {
    const auto b_mds = core::to_const_mdspan(std::forward<decltype(b)>(b));

    constexpr std::size_t rhs_rank = b_mds.rank() == 1 ? 1 : 2;

    core::batch_with_broadcast<backend>(
        [](auto &&a, auto &&b, auto &&x, auto &&valid) {
            valid() = ufunc::solve_ufunc(std::forward<decltype(a)>(a),
                                         std::forward<decltype(b)>(b),
                                         std::forward<decltype(x)>(x));
        },
        std::index_sequence<2, rhs_rank, rhs_rank, 0>{},
        std::integer_sequence<bool, true, true, false, false>{},
        std::forward<decltype(a)>(a), b_mds, std::forward<decltype(x)>(x),
        std::forward<decltype(valid)>(valid));
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto solve(auto &&a, auto &&b) {
    const auto a_mds = core::to_const_mdspan(std::forward<decltype(a)>(a));
    const auto b_mds = core::to_const_mdspan(std::forward<decltype(b)>(b));

    constexpr std::size_t rhs_rank = b_mds.rank() == 1 ? 1 : 2;

    auto x = core::make_broadcasted_tensor<dtype>(
        std::index_sequence<2, rhs_rank>{},
        core::slice_extents_from_right<rhs_rank>(b_mds.extents()), a_mds,
        b_mds);

    auto valid = core::make_broadcasted_tensor<bool>(
        std::index_sequence<2, rhs_rank>{}, core::extents<std::uint8_t>{},
        a_mds, b_mds);

    solve_to<backend>(a_mds, b_mds, x, valid);

    return std::pair{x, valid};
}

} // namespace mdtensor::linalg
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/linalg/solve.hpp

namespace mdtensor {

constexpr void matmul_to(auto &&...elems) {
    linalg::matmul_to(std::forward<decltype(elems)>(elems)...);
}

template <typename dtype = void>
[[nodiscard]] constexpr auto matmul(auto &&...elems) {
    return linalg::matmul<dtype>(std::forward<decltype(elems)>(elems)...);
}

constexpr void matvec_to(auto &&...elems) {
    linalg::matvec_to(std::forward<decltype(elems)>(elems)...);
}

template <typename dtype = void>
[[nodiscard]] constexpr auto matvec(auto &&...elems) {
    return linalg::matvec<dtype>(std::forward<decltype(elems)>(elems)...);
}

constexpr void vecmat_to(auto &&...elems) {
    linalg::vecmat_to(std::forward<decltype(elems)>(elems)...);
}

template <typename dtype = void>
[[nodiscard]] constexpr auto vecmat(auto &&...elems) {
    return linalg::vecmat<dtype>(std::forward<decltype(elems)>(elems)...);
}

} // namespace mdtensor
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


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/logical_and.hpp
/**
 * @file
 * @brief Element-wise logical AND utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace ufunc {

constexpr void logical_and_ufunc(auto &&in1, auto &&in2, auto &&out,
                                 auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = (static_cast<bool>(in1()) && static_cast<bool>(in2()));
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
logical_and(auto &&in1, auto &&in2, out_t &&out = out_t{std::nullopt},
            where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::logical_and_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/logical_and.hpp

namespace mdtensor {

template <typename dtype = bool, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO, std::integral axes_t,
          axes_t... axes, typename out_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto all(auto &&in,
                                 std::integer_sequence<axes_t, axes...>,
                                 out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_reduced_tensor<dtype, keepdims>(
                std::integer_sequence<axes_t, axes...>{},
                std::index_sequence<0>{}, core::extents<std::uint8_t>{},
                in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    // TODO: move to reduce
    fill<backend>(out_md, true);

    core::reduce<keepdims>(
        [](auto &&in, auto &&out, auto &&where) {
            static_cast<void>(logical_and<void, backend>(
                std::forward<decltype(in)>(in),
                std::forward<decltype(out)>(out),
                std::forward<decltype(out)>(out),
                std::forward<decltype(where)>(where)));
        },
        std::integer_sequence<axes_t, axes...>{},
        std::index_sequence<0, 0, 0>{},
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

template <std::int64_t axis, typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto all(auto &&in, out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return all<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in),
        std::integer_sequence<std::int64_t, axis>{},
        std::forward<decltype(out)>(out), std::forward<decltype(where)>(where));
}

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto all(auto &&in, out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return all<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in), std::index_sequence<>{},
        std::forward<decltype(out)>(out), std::forward<decltype(where)>(where));
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
namespace ufunc {

constexpr void isclose_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&rtol,
                             auto &&atol, const bool equal_nan) {
    if constexpr (requires {
                      { std::isnan(in1()) } -> std::convertible_to<bool>;
                      { std::isnan(in2()) } -> std::convertible_to<bool>;
                  }) {
        if (equal_nan) {
            if (std::isnan(in1()) && std::isnan(in2())) {
                out() = true;
                return;
            }
        }
    }

    // if both inputs are inf and same sign, return true (numpy-like)
    if constexpr (requires {
                      { std::isinf(in1()) } -> std::convertible_to<bool>;
                      { std::isinf(in2()) } -> std::convertible_to<bool>;
                      { std::signbit(in1()) } -> std::convertible_to<bool>;
                      { std::signbit(in2()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isinf(in1()) && std::isinf(in2()) &&
            std::signbit(in1()) == std::signbit(in2())) {
            out() = true;
            return;
        }
    }

    using out_t = std::remove_cvref_t<decltype(out())>;
    using calc_t = core::common_data_type_t<decltype(in1()), decltype(in2()),
                                            decltype(rtol()), decltype(atol())>;

    out() = static_cast<out_t>(
        absolute(static_cast<calc_t>(in1()) - static_cast<calc_t>(in2())) <=
        (static_cast<calc_t>(atol()) +
         static_cast<calc_t>(rtol()) * absolute(static_cast<calc_t>(in2()))));
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename rtol_t = double, typename atol_t = double,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto
isclose(auto &&in1, auto &&in2, rtol_t &&rtol = rtol_t{1e-05},
        atol_t &&atol = atol_t{1e-08}, out_t &&out = out_t{std::nullopt},
        const bool equal_nan = false) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto rtol_mds =
        core::to_const_mdspan(std::forward<decltype(rtol)>(rtol));
    const auto atol_mds =
        core::to_const_mdspan(std::forward<decltype(atol)>(atol));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds, rtol_mds,
                atol_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [&](auto &&...elems) {
            ufunc::isclose_ufunc(std::forward<decltype(elems)>(elems)...,
                                 equal_nan);
        },
        std::integer_sequence<bool, true, true, false, true, true>{}, in1_mds,
        in2_mds, out_md, rtol_mds, atol_mds);

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/isclose.hpp

namespace mdtensor {

template <core::Backend backend = core::Backend::AUTO, typename rtol_t = double,
          typename atol_t = double>
[[nodiscard]] constexpr bool
allclose(auto &&in1, auto &&in2, rtol_t &&rtol = rtol_t{1e-05},
         atol_t &&atol = atol_t{1e-08}, const bool equal_nan = false) {
    return all<void, false, core::Backend::NATIVE>(isclose<bool, backend>(
        std::forward<decltype(in1)>(in1), std::forward<decltype(in2)>(in2),
        std::forward<decltype(rtol)>(rtol), std::forward<decltype(atol)>(atol),
        std::nullopt, equal_nan));
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


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/logical_or.hpp
/**
 * @file
 * @brief Element-wise logical OR utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace ufunc {

constexpr void logical_or_ufunc(auto &&in1, auto &&in2, auto &&out,
                                auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = (static_cast<bool>(in1()) || static_cast<bool>(in2()));
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
logical_or(auto &&in1, auto &&in2, out_t &&out = out_t{std::nullopt},
           where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::logical_or_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/logical_or.hpp

namespace mdtensor {

template <typename dtype = bool, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO, std::integral axes_t,
          axes_t... axes, typename out_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto any(auto &&in,
                                 std::integer_sequence<axes_t, axes...>,
                                 out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_reduced_tensor<dtype, keepdims>(
                std::integer_sequence<axes_t, axes...>{},
                std::index_sequence<0>{}, core::extents<std::uint8_t>{},
                in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    // TODO: move to reduce
    fill<backend>(out_md, false);

    core::reduce<keepdims>(
        [](auto &&in, auto &&out, auto &&where) {
            static_cast<void>(logical_or<void, backend>(
                std::forward<decltype(in)>(in),
                std::forward<decltype(out)>(out),
                std::forward<decltype(out)>(out),
                std::forward<decltype(where)>(where)));
        },
        std::integer_sequence<axes_t, axes...>{},
        std::index_sequence<0, 0, 0>{},
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

template <std::int64_t axis, typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto any(auto &&in, out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return any<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in),
        std::integer_sequence<std::int64_t, axis>{},
        std::forward<decltype(out)>(out), std::forward<decltype(where)>(where));
}

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto any(auto &&in, out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return any<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in), std::index_sequence<>{},
        std::forward<decltype(out)>(out), std::forward<decltype(where)>(where));
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
namespace ufunc {

template <bool equal_nan>
constexpr bool array_equal_ufunc(auto &&in1, auto &&in2) {
    if constexpr (equal_nan && requires {
                      { std::isnan(in1()) } -> std::convertible_to<bool>;
                      { std::isnan(in2()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in1()) && std::isnan(in2())) {
            return true;
        }
    }

    using value_t = core::common_data_type_t<decltype(in1()), decltype(in2())>;

    return static_cast<value_t>(in1()) == static_cast<value_t>(in2());
}

} // namespace ufunc

namespace {

template <bool equal_nan>
[[nodiscard]] constexpr bool array_equal_impl_(auto &&in1, auto &&in2) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    if constexpr (in1_mds.rank() == 0) {
        return ufunc::array_equal_ufunc<equal_nan>(in1_mds, in2_mds);

    } else {
        using index_t = typename decltype(in1_mds)::index_type;

        for (index_t i = 0; i < in1_mds.extent(0); ++i) {
            if (!array_equal_impl_<equal_nan>(
                    core::submdspan_from_left(in1_mds, i),
                    core::submdspan_from_left(in2_mds, i))) {
                return false;
            }
        }

        return true;
    }
}

} // namespace

[[nodiscard]] constexpr bool array_equal(auto &&in1, auto &&in2,
                                         const bool equal_nan = false) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    if constexpr (in1_mds.rank() != in2_mds.rank()) {
        return false;

    } else {
        if (!core::is_same_extents(in1_mds.extents(), in2_mds.extents())) {
            return false;
        }

        if (equal_nan) {
            return array_equal_impl_<true>(in1_mds, in2_mds);

        } else {
            return array_equal_impl_<false>(in1_mds, in2_mds);
        }
    }
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



namespace mdtensor {

[[nodiscard]] constexpr bool array_equiv(auto &&in1, auto &&in2) {
    try {
        const auto [in1_bcast, in2_bcast] = std::get<0>(
            core::broadcast(std::index_sequence<0, 0>{},
                            std::integer_sequence<bool, true, true>{},
                            std::forward<decltype(in1)>(in1),
                            std::forward<decltype(in2)>(in2)));
        return array_equal(in1_bcast, in2_bcast);

    } catch (const std::exception &e) {
        return false;
    }
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/array_equiv.hpp
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
namespace ufunc {

constexpr void equal_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = (in1() == in2());
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto equal(auto &&in1, auto &&in2,
                                   out_t &&out = out_t{std::nullopt},
                                   where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::equal_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/equal.hpp
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
namespace ufunc {

constexpr void greater_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = (in1() > in2());
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto greater(auto &&in1, auto &&in2,
                                     out_t &&out = out_t{std::nullopt},
                                     where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::greater_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
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
namespace ufunc {

constexpr void greater_equal_ufunc(auto &&in1, auto &&in2, auto &&out,
                                   auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = (in1() >= in2());
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
greater_equal(auto &&in1, auto &&in2, out_t &&out = out_t{std::nullopt},
              where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::greater_equal_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/greater_equal.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/isinf.hpp
/**
 * @file
 * @brief Element-wise isinf check for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace ufunc {

constexpr void isinf_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    if constexpr (requires {
                      { std::isinf(in()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isinf(in())) {
            out() = true;
            return;
        }
    }

    out() = false;
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto isinf(auto &&in, out_t &&out = out_t{std::nullopt},
                                   where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return empty_like<dtype>(in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::isinf_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/isinf.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/isnan.hpp
/**
 * @file
 * @brief Element-wise isnan check for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace ufunc {

constexpr void isnan_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    if constexpr (requires {
                      { std::isnan(in()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in())) {
            out() = true;
            return;
        }
    }

    out() = false;
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto isnan(auto &&in, out_t &&out = out_t{std::nullopt},
                                   where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return empty_like<dtype>(in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::isnan_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/isnan.hpp
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
namespace ufunc {

constexpr void less_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = (in1() < in2());
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto less(auto &&in1, auto &&in2,
                                  out_t &&out = out_t{std::nullopt},
                                  where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::less_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
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
namespace ufunc {

constexpr void less_equal_ufunc(auto &&in1, auto &&in2, auto &&out,
                                auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = (in1() <= in2());
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
less_equal(auto &&in1, auto &&in2, out_t &&out = out_t{std::nullopt},
           where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::less_equal_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/less_equal.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/logical_not.hpp
/**
 * @file
 * @brief Element-wise logical NOT utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace ufunc {

constexpr void logical_not_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = (!static_cast<bool>(in()));
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
logical_not(auto &&in, out_t &&out = out_t{std::nullopt},
            where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return empty_like<dtype>(in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::logical_not_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/logical_not.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/logical_xor.hpp
/**
 * @file
 * @brief Element-wise logical XOR utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace ufunc {

constexpr void logical_xor_ufunc(auto &&in1, auto &&in2, auto &&out,
                                 auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = (static_cast<bool>(in1()) != static_cast<bool>(in2()));
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
logical_xor(auto &&in1, auto &&in2, out_t &&out = out_t{std::nullopt},
            where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::logical_xor_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/logic/logical_xor.hpp
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
namespace ufunc {

constexpr void not_equal_ufunc(auto &&in1, auto &&in2, auto &&out,
                               auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = (in1() != in2());
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
not_equal(auto &&in1, auto &&in2, out_t &&out = out_t{std::nullopt},
          where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::not_equal_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
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


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/broadcast.hpp
/**
 * @file
 * @brief Broadcast utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

[[nodiscard]] constexpr auto broadcast(auto &&...ins) {
    return std::get<0>([&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return core::broadcast(
            std::index_sequence<((void)Is, 0)...>{},
            std::integer_sequence<bool, (void(Is), true)...>{},
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{}));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/broadcast.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/broadcast_to.hpp
/**
 * @file
 * @brief Broadcast_to utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

[[nodiscard]] constexpr auto broadcast_to(auto &&in, auto &&shape) {
    return core::broadcast_to(std::forward<decltype(in)>(in),
                              std::forward<decltype(shape)>(shape));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/broadcast_to.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/concatenate.hpp
/**
 * @file
 * @brief Concatenation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {
namespace detail {

template <bool concatenate, std::size_t... Extents>
[[nodiscard]] consteval std::size_t concatenate_static_extent() noexcept {
    static_assert(sizeof...(Extents) > 0,
                  "At least one extent must be provided for concatenation.");

    if constexpr (concatenate) {
        if constexpr (((Extents == core::dyn) || ...)) {
            return core::dyn;

        } else {
            return (Extents + ...);
        }

    } else {
        if constexpr (((Extents == core::dyn) && ...)) {
            return core::dyn;

        } else {
            constexpr std::size_t cext =
                std::max({((Extents != core::dyn) ? Extents : 0)...});

            static_assert(((Extents == cext || Extents == core::dyn) && ...),
                          "Incompatible static extents for concatenation.");

            return cext;
        }
    }
}

template <bool concatenate, typename index_t,
          std::convertible_to<index_t>... exts_t>
[[nodiscard]] constexpr index_t concatenate_extent(exts_t &&...exts) {
    static_assert(sizeof...(exts) > 0,
                  "At least one extent must be provided for concatenation.");

    if constexpr (concatenate) {
        return (exts + ...);

    } else {
        const index_t cext = std::get<0>(std::forward_as_tuple(exts...));

        if (((cext != static_cast<index_t>(exts)) && ...)) {
            throw std::invalid_argument(
                "Incompatible extents for concatenation.");
        }

        return cext;
    }
}

template <std::int64_t axis, core::extents_c... ins_t>
[[nodiscard]] constexpr auto concatenate_extents(ins_t &&...ins) {
    static_assert(sizeof...(ins) > 0,
                  "At least one extents must be provided for concatenation.");

    using index_t = core::common_index_type_t<
        typename std::remove_cvref_t<ins_t>::index_type...>;

    constexpr std::size_t rank = std::remove_cvref_t<
        std::tuple_element_t<0, std::tuple<ins_t...>>>::rank();

    static_assert(((ins.rank() == rank) && ...),
                  "All input extents must have the same rank.");

    if constexpr (rank == 0) {
        return core::extents<index_t>{};

    } else {
        constexpr std::size_t baxis =
            static_cast<std::size_t>(core::bounding_index(axis, rank - 1));

        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            const auto static_extent_at = [&]<std::size_t I>() {
                return concatenate_static_extent<
                    I == baxis,
                    std::remove_cvref_t<ins_t>::static_extent(I)...>();
            };

            const auto extent_at = [&]<std::size_t I>() {
                return concatenate_extent<I == baxis, index_t>(
                    ins.extent(I)...);
            };

            return core::extents<
                index_t, static_extent_at.template operator()<Is>()...> {
                extent_at.template operator()<Is>()...
            };
        }(std::make_index_sequence<rank>{});
    }
}

} // namespace detail

template <std::int64_t axis = 0, typename dtype = void>
[[nodiscard]] constexpr auto concatenate(auto &&...ins) {
    if constexpr ((!core::mdspan_c<decltype(ins)> || ...)) {
        return concatenate<axis>(
            core::to_const_mdspan(std::forward<decltype(ins)>(ins))...);

    } else {
        constexpr std::size_t rank = std::remove_cvref_t<
            std::tuple_element_t<0, std::tuple<decltype(ins)...>>>::rank();
        constexpr std::size_t baxis =
            static_cast<std::size_t>(core::bounding_index(axis, rank - 1));

        // generate out extents
        const auto out_extents =
            detail::concatenate_extents<axis>(ins.extents()...);

        // generate out
        using value_t = core::output_value_t<
            dtype, typename std::remove_cvref_t<decltype(ins)>::value_type...>;
        auto out = empty<value_t>(out_extents);

        // concatenate
        using index_t = typename decltype(out_extents)::index_type;
        index_t offset = 0;

        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&] {
                 const auto in = std::get<Is>(std::forward_as_tuple(
                     std::forward<decltype(ins)>(ins)...));

                 const index_t extent = static_cast<index_t>(in.extent(baxis));

                 static_cast<void>(copy(in, core::submdspan_from_left<baxis>(
                                                out, core::stdex::strided_slice{
                                                         offset, extent, 1})));

                 offset += extent;
             })(),
             ...);
        }(std::make_index_sequence<sizeof...(ins)>{});

        return out;
    }
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/concatenate.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/flatten.hpp
/**
 * @file
 * @brief Flatten utilities for mdtensor.
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

template <core::Copy copy = core::Copy::AUTO>
[[nodiscard]] constexpr auto reshape(auto &&in, auto &&shape) {
    return core::reshape<copy>(std::forward<decltype(in)>(in),
                               std::forward<decltype(shape)>(shape));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/reshape.hpp

namespace mdtensor {
namespace detail {

template <std::size_t... Extents>
[[nodiscard]] consteval std::size_t flatten_static_extent() noexcept {
    static_assert(sizeof...(Extents) > 0,
                  "At least one extent must be provided for flattening.");

    if constexpr (((Extents == core::dyn) || ...)) {
        return core::dyn;

    } else {
        return (Extents * ...);
    }
}

template <typename index_t, std::convertible_to<index_t>... exts_t>
[[nodiscard]] constexpr index_t flatten_extent(exts_t &&...exts) noexcept {
    static_assert(sizeof...(exts) > 0,
                  "At least one extent must be provided for flattening.");

    return (exts * ...);
}

template <core::extents_c in_t>
[[nodiscard]] constexpr auto flatten_extents(in_t &&in) noexcept {
    using base_t = std::remove_cvref_t<in_t>;
    using index_t = typename base_t::index_type;

    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return core::extents<
            typename base_t::index_type,
            flatten_static_extent<base_t::static_extent(Is)...>()>{
            flatten_extent<index_t>(in.extent(Is)...)};
    }(std::make_index_sequence<base_t::rank()>{});
}

} // namespace detail

[[nodiscard]] constexpr auto flatten(auto &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    return reshape<core::Copy::TRUE>(in_mds,
                                     detail::flatten_extents(in_mds.extents()));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/flatten.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/manipulation/transpose.hpp
/**
 * @file
 * @brief Transpose utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

template <std::integral axes_t, axes_t... axes>
[[nodiscard]] constexpr auto transpose(auto &&in,
                                       std::integer_sequence<axes_t, axes...>) {
    const auto in_mds = core::to_mdspan(std::forward<decltype(in)>(in));
    using in_mds_t = decltype(in_mds);

    constexpr std::size_t rank = in_mds_t::rank();

    if constexpr (rank < 2) {
        return in_mds;

    } else if constexpr (sizeof...(axes) == 0) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return transpose(std::forward<decltype(in)>(in),
                             std::index_sequence<(rank - 1 - Is)...>{});
        }(std::make_index_sequence<rank>{});

    } else {
        static_assert(sizeof...(axes) == rank,
                      "Number of axes must match rank.");

        constexpr auto axes_arr = std::array{static_cast<std::size_t>(
            core::bounding_index<axes_t>(axes, rank - 1))...};

        static_assert(
            [&]() {
                for (std::size_t i = 0; i < axes_arr.size(); i++) {
                    for (std::size_t j = i + 1; j < axes_arr.size(); j++) {
                        if (axes_arr[i] == axes_arr[j]) {
                            return false;
                        }
                    }
                }
                return true;
            }(),
            "Axes must be unique.");

        const auto new_extents =
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return core::extents<typename in_mds_t::index_type,
                                     in_mds_t::static_extent(axes_arr[Is])...>{
                    in_mds.extent(axes_arr[Is])...};
            }(std::make_index_sequence<axes_arr.size()>{});

        const auto new_strides =
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::array<typename in_mds_t::index_type, rank>{
                    in_mds.stride(axes_arr[Is])...};
            }(std::make_index_sequence<axes_arr.size()>{});

        return core::mdspan<typename in_mds_t::element_type,
                            std::remove_cvref_t<decltype(new_extents)>,
                            core::stdex::layout_stride,
                            typename in_mds_t::accessor_type>{
            in_mds.data_handle(),
            core::stdex::layout_stride::mapping{new_extents, new_strides}};
    }
}

template <std::int64_t... axes>
[[nodiscard]] constexpr auto transpose(auto &&in) {
    return transpose(std::forward<decltype(in)>(in),
                     std::integer_sequence<std::int64_t, axes...>{});
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
namespace ufunc {

constexpr void atan2_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    using value_t = core::common_data_type_t<decltype(in1()), decltype(in2()),
                                             decltype(out())>;

    out() =
        std::atan2(static_cast<value_t>(in1()), static_cast<value_t>(in2()));
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto atan2(auto &&in1, auto &&in2,
                                   out_t &&out = out_t{std::nullopt},
                                   where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            // NOTE: ensure that the output type is at least float precision
            using value_t = core::output_value_t<
                dtype, typename decltype(in1_mds)::value_type,
                typename decltype(in2_mds)::value_type, float>;

            return core::make_broadcasted_tensor<value_t>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::atan2_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
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
namespace ufunc {

constexpr void clip_ufunc(auto &&in, auto &&min, auto &&max, auto &&out) {
    // NOTE: std::clamp is not used to match the behavior with original np.clip
    // when min > max, np.clip returns max, and std::clamp returns min.
    // mdtensor.clip is designed to match the behavior of np.clip.

    using value_t = std::remove_cvref_t<decltype(in())>;

    out() = in();

    if constexpr (!core::nullopt_t_c<decltype(min())>) {
        out() = std::max(out(), static_cast<value_t>(min()));
    }

    if constexpr (!core::nullopt_t_c<decltype(max())>) {
        out() = std::min(out(), static_cast<value_t>(max()));
    }
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename min_t = std::nullopt_t, typename max_t = std::nullopt_t,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto clip(auto &&in, min_t &&min = min_t{std::nullopt},
                                  max_t &&max = max_t{std::nullopt},
                                  out_t &&out = out_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return empty_like<dtype>(in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::clip_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, true, false>{}, in_mds,
        std::forward<decltype(min)>(min), std::forward<decltype(max)>(max),
        out_md);

    return out_md;
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
namespace ufunc {

constexpr void cos_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    using value_t = core::common_data_type_t<decltype(in()), decltype(out())>;

    out() = std::cos(static_cast<value_t>(in()));
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto cos(auto &&in, out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            // NOTE: ensure that the output type is at least float precision
            using value_t = core::output_value_t<
                dtype, typename decltype(in_mds)::value_type, float>;

            return core::make_broadcasted_tensor<value_t>(
                core::extents<std::uint8_t>{}, in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::cos_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
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



namespace mdtensor {

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto deg2rad(auto &&in,
                                     out_t &&out = out_t{std::nullopt},
                                     where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            // NOTE: ensure that the output type is at least float precision
            using value_t = core::output_value_t<
                dtype, typename decltype(in_mds)::value_type, float>;

            return core::make_broadcasted_tensor<value_t>(
                core::extents<std::uint8_t>{}, in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    using calc_t = core::common_data_type_t<
        typename decltype(in_mds)::value_type,
        typename core::to_mdspan_t<decltype(out_md)>::value_type>;

    static_assert(std::is_floating_point_v<calc_t> &&
                  "deg2rad conversion requires at least float precision.");

    constexpr calc_t D2R = std::numbers::pi_v<calc_t> / calc_t{180};

    static_cast<void>(multiply<void, backend>(
        in_mds, D2R, out_md, std::forward<decltype(where)>(where)));

    return out_md;
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
namespace ufunc {

constexpr void divide_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = in1() / in2();
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto divide(auto &&in1, auto &&in2,
                                    out_t &&out = out_t{std::nullopt},
                                    where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::divide_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
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
namespace ufunc {

constexpr void maximum_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    using value_t = std::remove_cvref_t<decltype(out())>;

    // if one of the inputs is NaN, return NaN (numpy-like)
    if constexpr (requires {
                      { std::isnan(in1()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in1())) {
            out() = std::numeric_limits<value_t>::quiet_NaN();
            return;
        }
    }

    if constexpr (requires {
                      { std::isnan(in2()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in2())) {
            out() = std::numeric_limits<value_t>::quiet_NaN();
            return;
        }
    }

    out() = std::max(static_cast<value_t>(in1()), static_cast<value_t>(in2()));
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto maximum(auto &&in1, auto &&in2,
                                     out_t &&out = out_t{std::nullopt},
                                     where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::maximum_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/maximum.hpp

namespace mdtensor {

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO, std::integral axes_t,
          axes_t... axes, typename out_t = std::nullopt_t,
          typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto max(auto &&in,
                                 std::integer_sequence<axes_t, axes...>,
                                 out_t &&out = out_t{std::nullopt},
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_reduced_tensor<dtype, keepdims>(
                std::integer_sequence<axes_t, axes...>{},
                std::index_sequence<0>{}, core::extents<std::uint8_t>{},
                in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    // TODO: move to reduce
    if constexpr (core::nullopt_t_c<decltype(initial)>) {
        using value_t =
            typename core::to_mdspan_t<decltype(out_md)>::value_type;

        fill<backend>(out_md, std::numeric_limits<value_t>::lowest());

    } else {
        fill<backend>(out_md, std::forward<decltype(initial)>(initial));
    }

    core::reduce<keepdims>(
        [](auto &&in, auto &&out, auto &&where) {
            static_cast<void>(
                maximum<void, backend>(std::forward<decltype(in)>(in),
                                       std::forward<decltype(out)>(out),
                                       std::forward<decltype(out)>(out),
                                       std::forward<decltype(where)>(where)));
        },
        std::integer_sequence<axes_t, axes...>{},
        std::index_sequence<0, 0, 0>{},
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

template <std::int64_t axis, typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto max(auto &&in, out_t &&out = out_t{std::nullopt},
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return max<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in),
        std::integer_sequence<std::int64_t, axis>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial),
        std::forward<decltype(where)>(where));
}

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto max(auto &&in, out_t &&out = out_t{std::nullopt},
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return max<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in), std::index_sequence<>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial),
        std::forward<decltype(where)>(where));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/max.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/min.hpp
/**
 * @file
 * @brief Minimum reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


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
namespace ufunc {

constexpr void minimum_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    using value_t = std::remove_cvref_t<decltype(out())>;

    // if one of the inputs is NaN, return NaN (numpy-like)
    if constexpr (requires {
                      { std::isnan(in1()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in1())) {
            out() = std::numeric_limits<value_t>::quiet_NaN();
            return;
        }
    }

    if constexpr (requires {
                      { std::isnan(in2()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in2())) {
            out() = std::numeric_limits<value_t>::quiet_NaN();
            return;
        }
    }

    out() = std::min(static_cast<value_t>(in1()), static_cast<value_t>(in2()));
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto minimum(auto &&in1, auto &&in2,
                                     out_t &&out = out_t{std::nullopt},
                                     where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::extents<std::uint8_t>{}, in1_mds, in2_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::minimum_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/minimum.hpp

namespace mdtensor {

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO, std::integral axes_t,
          axes_t... axes, typename out_t = std::nullopt_t,
          typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto min(auto &&in,
                                 std::integer_sequence<axes_t, axes...>,
                                 out_t &&out = out_t{std::nullopt},
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_reduced_tensor<dtype, keepdims>(
                std::integer_sequence<axes_t, axes...>{},
                std::index_sequence<0>{}, core::extents<std::uint8_t>{},
                in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    // TODO: move to reduce
    if constexpr (core::nullopt_t_c<decltype(initial)>) {
        using value_t =
            typename core::to_mdspan_t<decltype(out_md)>::value_type;

        fill<backend>(out_md, std::numeric_limits<value_t>::max());

    } else {
        fill<backend>(out_md, std::forward<decltype(initial)>(initial));
    }

    core::reduce<keepdims>(
        [](auto &&in, auto &&out, auto &&where) {
            static_cast<void>(
                minimum<void, backend>(std::forward<decltype(in)>(in),
                                       std::forward<decltype(out)>(out),
                                       std::forward<decltype(out)>(out),
                                       std::forward<decltype(where)>(where)));
        },
        std::integer_sequence<axes_t, axes...>{},
        std::index_sequence<0, 0, 0>{},
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

template <std::int64_t axis, typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto min(auto &&in, out_t &&out = out_t{std::nullopt},
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return min<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in),
        std::integer_sequence<std::int64_t, axis>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial),
        std::forward<decltype(where)>(where));
}

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto min(auto &&in, out_t &&out = out_t{std::nullopt},
                                 initial_t &&initial = initial_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    return min<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in), std::index_sequence<>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial),
        std::forward<decltype(where)>(where));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/min.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/nanmax.hpp
/**
 * @file
 * @brief Maximum reduction without NaN utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO, std::integral axes_t,
          axes_t... axes, typename out_t = std::nullopt_t,
          typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
nanmax(auto &&in, std::integer_sequence<axes_t, axes...>,
       out_t &&out = out_t{std::nullopt},
       initial_t &&initial = initial_t{std::nullopt},
       where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto where_mds =
        core::to_const_mdspan(std::forward<decltype(where)>(where));

    const auto is_not_nan = logical_not(isnan(in_mds, std::nullopt, where_mds));

    const auto mask = [&]() {
        if constexpr (core::nullopt_t_c<decltype(where)>) {
            return is_not_nan;

        } else {
            return logical_and(is_not_nan, where_mds);
        }
    }();

    return max<dtype, keepdims, backend>(
        in_mds, std::integer_sequence<axes_t, axes...>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial), mask);
}

template <std::int64_t axis, typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
nanmax(auto &&in, out_t &&out = out_t{std::nullopt},
       initial_t &&initial = initial_t{std::nullopt},
       where_t &&where = where_t{std::nullopt}) {
    return nanmax<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in),
        std::integer_sequence<std::int64_t, axis>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial),
        std::forward<decltype(where)>(where));
}

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
nanmax(auto &&in, out_t &&out = out_t{std::nullopt},
       initial_t &&initial = initial_t{std::nullopt},
       where_t &&where = where_t{std::nullopt}) {
    return nanmax<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in), std::index_sequence<>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial),
        std::forward<decltype(where)>(where));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/nanmax.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/nanmin.hpp
/**
 * @file
 * @brief Minimum reduction without NaN utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO, std::integral axes_t,
          axes_t... axes, typename out_t = std::nullopt_t,
          typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
nanmin(auto &&in, std::integer_sequence<axes_t, axes...>,
       out_t &&out = out_t{std::nullopt},
       initial_t &&initial = initial_t{std::nullopt},
       where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto where_mds =
        core::to_const_mdspan(std::forward<decltype(where)>(where));

    const auto is_not_nan = logical_not(isnan(in_mds, std::nullopt, where_mds));

    const auto mask = [&]() {
        if constexpr (core::nullopt_t_c<decltype(where)>) {
            return is_not_nan;

        } else {
            return logical_and(is_not_nan, where_mds);
        }
    }();

    return min<dtype, keepdims, backend>(
        in_mds, std::integer_sequence<axes_t, axes...>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial), mask);
}

template <std::int64_t axis, typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
nanmin(auto &&in, out_t &&out = out_t{std::nullopt},
       initial_t &&initial = initial_t{std::nullopt},
       where_t &&where = where_t{std::nullopt}) {
    return nanmin<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in),
        std::integer_sequence<std::int64_t, axis>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial),
        std::forward<decltype(where)>(where));
}

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
nanmin(auto &&in, out_t &&out = out_t{std::nullopt},
       initial_t &&initial = initial_t{std::nullopt},
       where_t &&where = where_t{std::nullopt}) {
    return nanmin<dtype, keepdims, backend>(
        std::forward<decltype(in)>(in), std::index_sequence<>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial),
        std::forward<decltype(where)>(where));
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/nanmin.hpp
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
namespace ufunc {

constexpr void negative_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = -in();
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto negative(auto &&in,
                                      out_t &&out = out_t{std::nullopt},
                                      where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return empty_like<dtype>(in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::negative_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
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



namespace mdtensor {

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto rad2deg(auto &&in,
                                     out_t &&out = out_t{std::nullopt},
                                     where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            // NOTE: ensure that the output type is at least float precision
            using value_t = core::output_value_t<
                dtype, typename decltype(in_mds)::value_type, float>;

            return core::make_broadcasted_tensor<value_t>(
                core::extents<std::uint8_t>{}, in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    using calc_t = core::common_data_type_t<
        typename decltype(in_mds)::value_type,
        typename core::to_mdspan_t<decltype(out_md)>::value_type>;

    static_assert(std::is_floating_point_v<calc_t> &&
                  "rad2deg conversion requires at least float precision.");

    constexpr calc_t R2D = std::numbers::inv_pi_v<calc_t> * calc_t{180};

    static_cast<void>(multiply<void, backend>(
        in_mds, R2D, out_md, std::forward<decltype(where)>(where)));

    return out_md;
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
namespace ufunc {

constexpr void sign_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    out() = (in() > 0) - (in() < 0);
}

} // namespace ufunc

template <typename dtype = std::int8_t,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto sign(auto &&in, out_t &&out = out_t{std::nullopt},
                                  where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return empty_like<dtype>(in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::sign_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
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
namespace ufunc {

constexpr void sin_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    using value_t = core::common_data_type_t<decltype(in()), decltype(out())>;

    out() = std::sin(static_cast<value_t>(in()));
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto sin(auto &&in, out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            // NOTE: ensure that the output type is at least float precision
            using value_t = core::output_value_t<
                dtype, typename decltype(in_mds)::value_type, float>;

            return core::make_broadcasted_tensor<value_t>(
                core::extents<std::uint8_t>{}, in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::sin_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/math/sin.hpp
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
namespace ufunc {

constexpr void tan_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    using value_t = core::common_data_type_t<decltype(in()), decltype(out())>;

    out() = std::tan(static_cast<value_t>(in()));
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto tan(auto &&in, out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            // NOTE: ensure that the output type is at least float precision
            using value_t = core::output_value_t<
                dtype, typename decltype(in_mds)::value_type, float>;

            return core::make_broadcasted_tensor<value_t>(
                core::extents<std::uint8_t>{}, in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::tan_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
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


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/generator/generator.hpp
/**
 * @file
 * @brief Random generator header aggregator for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/generator/splitmix64.hpp
/**
 * @file
 * @brief SplitMix64 random number generator for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


#include <cstdint>

namespace mdtensor::random::generator {

class SplitMix64 {
  public:
    using result_type = std::uint64_t;

  public:
    constexpr explicit SplitMix64(result_type seed) noexcept : state_(seed) {}

  public:
    [[nodiscard]] constexpr result_type operator()() noexcept {
        // https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64

        result_type z = (state_ += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;

        return z ^ (z >> 31);
    }

  private:
    result_type state_;
};

} // namespace mdtensor::random::generator
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/generator/splitmix64.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/generator/wrapper.hpp
/**
 * @file
 * @brief Random number generator engine wrapper for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


#include <cstdint>
#include <limits>
#include <random>

namespace mdtensor::random::generator {

template <typename EngineType = std::mt19937_64> class EngineWrapper {
  public:
    using engine_t = std::remove_cvref_t<EngineType>;
    using base_t = typename engine_t::result_type;

  public:
    constexpr explicit EngineWrapper() : engine_(engine_t{}) {}
    constexpr explicit EngineWrapper(const base_t seed)
        : engine_(engine_t{seed}) {}

  public:
    [[nodiscard]] constexpr auto operator()() { return engine_(); }

    template <std::integral value_t> [[nodiscard]] constexpr value_t get() {
        using int_t = std::remove_cvref_t<value_t>;

        if constexpr (std::same_as<int_t, bool>) {
            return ((*this)() >> (std::numeric_limits<base_t>::digits - 1)) !=
                   0;

        } else {
            using uint_t = std::make_unsigned_t<int_t>;

            constexpr std::size_t base_bits =
                std::numeric_limits<base_t>::digits;
            constexpr std::size_t uint_bits =
                std::numeric_limits<uint_t>::digits;

            if constexpr (base_bits >= uint_bits) {
                const base_t raw = (*this)();
                const uint_t bits =
                    static_cast<uint_t>(raw >> (base_bits - uint_bits));

                return from_ordered_unsigned<int_t>(bits);

            } else {
                uint_t bits = 0;
                std::size_t shift = 0;

                while (shift < uint_bits) {
                    bits |= static_cast<uint_t>((*this)()) << shift;
                    shift += base_bits;
                }

                return from_ordered_unsigned<int_t>(bits);
            }
        }
    }

    template <std::unsigned_integral value_t>
    [[nodiscard]] constexpr value_t get_bounded(const value_t bound) {
        using uint_t = std::remove_cvref_t<value_t>;

        if (bound == 0) {
            assert(false && "get_bounded: bound must be greater than 0");
            return 0;

        } else if (std::same_as<uint_t, bool>) {
            return get<bool>();

        } else {
            const uint_t thold = static_cast<uint_t>(uint_t{0} - bound) % bound;

            while (true) {
                const uint_t value = get<uint_t>();

                if (value >= thold) {
                    return value % bound;
                }
            }
        }
    }

    template <std::integral value_t>
    [[nodiscard]] constexpr value_t get_bounded(const value_t low,
                                                const value_t high) {
        using int_t = std::remove_cvref_t<value_t>;

        if (low >= high) {
            assert(false && "get_bounded: low must be less than high");

            // NOTE: select mdtensor::clip like behavior
            return high;

        } else if (std::same_as<int_t, bool>) {
            return get<bool>() ? high : low;

        } else {
            using uint_t = std::make_unsigned_t<int_t>;

            const uint_t u_low = to_ordered_unsigned<int_t>(low);
            const uint_t u_high = to_ordered_unsigned<int_t>(high);

            const uint_t bound = u_high - u_low;

            return from_ordered_unsigned<int_t>(u_low +
                                                get_bounded<uint_t>(bound));
        }
    }

  private:
    template <std::integral value_t>
    [[nodiscard]] static constexpr std::make_unsigned_t<value_t>
    to_ordered_unsigned(const value_t value) {
        using int_t = std::remove_cvref_t<value_t>;

        if constexpr (std::is_unsigned_v<int_t>) {
            return value;

        } else {
            using sint_t = int_t;
            using uint_t = std::make_unsigned_t<sint_t>;

            constexpr uint_t begin =
                static_cast<uint_t>(std::numeric_limits<sint_t>::max()) +
                uint_t{1};

            if (value < sint_t{0}) {
                return static_cast<uint_t>(
                    value - std::numeric_limits<sint_t>::lowest());
            }

            return static_cast<uint_t>(begin + static_cast<uint_t>(value));
        }
    }

    template <std::integral value_t>
    [[nodiscard]] static constexpr auto
    from_ordered_unsigned(const std::make_unsigned_t<value_t> value) {
        using int_t = std::remove_cvref_t<value_t>;

        if constexpr (std::is_unsigned_v<int_t>) {
            return value;

        } else {
            using sint_t = int_t;
            using uint_t = std::make_unsigned_t<sint_t>;

            constexpr uint_t begin =
                static_cast<uint_t>(std::numeric_limits<sint_t>::max()) +
                uint_t{1};

            if (value < begin) {
                return static_cast<sint_t>(
                    std::numeric_limits<sint_t>::lowest() +
                    static_cast<sint_t>(value));

            } else {
                return static_cast<sint_t>(value - begin);
            }
        }
    }

  private:
    engine_t engine_;
};

} // namespace mdtensor::random::generator
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/generator/wrapper.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/generator/generator.hpp

//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/rand.hpp
/**
 * @file
 * @brief Random number generation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/randint.hpp
/**
 * @file
 * @brief Random integer generation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


#include <random>

//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/seed.hpp
/**
 * @file
 * @brief Random seed utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


#include <random>


namespace mdtensor::random {

struct seed_t {
  public:
    using value_t = std::uint64_t;

  public:
    value_t value;
};

[[nodiscard]] inline seed_t make_random_seed() {
    auto engine = generator::EngineWrapper<std::random_device>{};

    return seed_t{engine.template get<typename seed_t::value_t>()};
}

} // namespace mdtensor::random
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/seed.hpp

namespace mdtensor::random {

using default_random_engine_t = std::mt19937_64;

namespace ufunc {

constexpr void randint_ufunc(auto &&out, auto &&low, auto &&high,
                             auto &&engine) {
    using value_t = std::remove_cvref_t<decltype(out())>;

    static_assert(std::is_integral_v<value_t>,
                  "randint_ufunc requires integral value type.");

    constexpr bool has_low =
        !core::nullopt_t_c<std::remove_cvref_t<decltype(low())>>;
    constexpr bool has_high =
        !core::nullopt_t_c<std::remove_cvref_t<decltype(high())>>;

    if constexpr (has_low && has_high) {
        // NOTE: This implementation matches the behavior of
        // numpy.random.randint(low, high)

        out() = engine.template get_bounded<value_t>(
            static_cast<value_t>(low()), static_cast<value_t>(high()));

    } else if constexpr (has_low && !has_high) {
        // NOTE: This implementation matches the behavior of
        // numpy.random.randint(low, high=None)

        out() = engine.template get_bounded<value_t>(
            value_t{0}, static_cast<value_t>(low()));

    } else if constexpr (!has_low && has_high) {
        // NOTE: This implementation is not exist in numpy.random.randint,
        // but maybe useful for some use cases.

        out() = engine.template get_bounded<value_t>(
            std::numeric_limits<value_t>::lowest(),
            static_cast<value_t>(high()));

    } else {
        // NOTE: This implementation is not exist in numpy.random.randint,
        // but maybe useful for some use cases.

        out() = engine.template get<value_t>();
    }
}

} // namespace ufunc

template <typename dtype = int, typename EngineType = default_random_engine_t,
          typename shape_t = core::extents<std::uint8_t>,
          typename low_t = std::nullopt_t, typename high_t = std::nullopt_t,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto
randint(shape_t &&shape = shape_t{}, low_t &&low = low_t{std::nullopt},
        high_t &&high = high_t{std::nullopt}, out_t &&out = out_t{std::nullopt},
        const seed_t seed = make_random_seed()) {
    const auto low_mds =
        core::to_const_mdspan(std::forward<decltype(low)>(low));
    const auto high_mds =
        core::to_const_mdspan(std::forward<decltype(high)>(high));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::to_extents(std::forward<decltype(shape)>(shape)), low_mds,
                high_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    static_assert(core::integral_c<
                      typename core::to_mdspan_t<decltype(out_md)>::value_type>,
                  "Output must have a integral value type.");

    auto engine = generator::EngineWrapper<EngineType>{seed.value};

    core::batch_with_broadcast<core::Backend::NATIVE>(
        [&](auto &&...elems) {
            ufunc::randint_ufunc(std::forward<decltype(elems)>(elems)...,
                                 engine);
        },
        std::integer_sequence<bool, false, true, true>{}, out_md, low_mds,
        high_mds);

    return out_md;
}

} // namespace mdtensor::random
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/randint.hpp

namespace mdtensor::random {
namespace ufunc {

template <std::floating_point value_t>
[[nodiscard]] constexpr value_t pow2_neg(std::size_t bits) noexcept {
    value_t result = value_t{1};

    for (std::size_t i = 0; i < bits; i++) {
        result *= value_t{0.5};
    }

    return result;
}

constexpr void rand_ufunc(auto &&out, auto &&engine) {
    using value_t = std::remove_cvref_t<decltype(out())>;

    static_assert(std::is_floating_point_v<value_t>,
                  "rand_ufunc requires floating-point value type.");

    using base_t = typename std::remove_cvref_t<decltype(engine)>::base_t;

    static_assert(std::unsigned_integral<base_t>,
                  "rand_ufunc requires an unsigned integral engine result.");

    constexpr std::size_t base_bits = std::numeric_limits<base_t>::digits;
    constexpr std::size_t value_bits = std::numeric_limits<value_t>::digits;

    if constexpr (value_bits <= base_bits) {
        const base_t bits = engine() >> (base_bits - value_bits);
        out() = static_cast<value_t>(bits) * pow2_neg<value_t>(value_bits);

    } else {
        value_t result = value_t{0};
        value_t scale = value_t{1};

        std::size_t remaining = value_bits;

        while (remaining > 0) {
            const std::size_t take =
                remaining < base_bits ? remaining : base_bits;

            const base_t bits = engine() >> (base_bits - take);

            scale *= pow2_neg<value_t>(take);
            result += static_cast<value_t>(bits) * scale;

            remaining -= take;
        }

        out() = result;
    }
}

} // namespace ufunc

template <typename dtype = double,
          typename EngineType = default_random_engine_t,
          typename shape_t = core::extents<std::uint8_t>,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto rand(shape_t &&shape = shape_t{},
                                  out_t &&out = out_t{std::nullopt},
                                  const seed_t seed = make_random_seed()) {
    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return empty<dtype>(std::forward<decltype(shape)>(shape));

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    static_assert(core::floating_point_c<
                      typename core::to_mdspan_t<decltype(out_md)>::value_type>,
                  "Output must have a floating point value type.");

    auto engine = generator::EngineWrapper<EngineType>{seed.value};

    core::batch<core::Backend::NATIVE,
                core::to_mdspan_t<decltype(out_md)>::rank()>(
        [&](auto &&...elems) {
            ufunc::rand_ufunc(std::forward<decltype(elems)>(elems)..., engine);
        },
        out_md);

    return out_md;
}

} // namespace mdtensor::random
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



namespace mdtensor::random {
namespace ufunc {

constexpr void uniform_ufunc(auto &&out, auto &&low, auto &&high,
                             auto &&engine) {
    using value_t = std::remove_cvref_t<decltype(out())>;

    rand_ufunc(std::forward<decltype(out)>(out),
               std::forward<decltype(engine)>(engine));

    out() =
        (static_cast<value_t>(high()) - static_cast<value_t>(low())) * out() +
        static_cast<value_t>(low());
}

} // namespace ufunc

template <
    typename dtype = double, typename EngineType = default_random_engine_t,
    typename shape_t = core::extents<std::uint8_t>, typename low_t = dtype,
    typename high_t = dtype, typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto
uniform(shape_t &&shape = shape_t{}, low_t &&low = low_t{0},
        high_t &&high = high_t{1}, out_t &&out = out_t{std::nullopt},
        const seed_t seed = make_random_seed()) {
    const auto low_mds =
        core::to_const_mdspan(std::forward<decltype(low)>(low));
    const auto high_mds =
        core::to_const_mdspan(std::forward<decltype(high)>(high));

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_broadcasted_tensor<dtype>(
                core::to_extents(std::forward<decltype(shape)>(shape)), low_mds,
                high_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    static_assert(core::floating_point_c<
                      typename core::to_mdspan_t<decltype(out_md)>::value_type>,
                  "Output must have a floating point value type.");

    auto engine = generator::EngineWrapper<EngineType>{seed.value};

    core::batch_with_broadcast<core::Backend::NATIVE>(
        [&](auto &&...elems) {
            ufunc::uniform_ufunc(std::forward<decltype(elems)>(elems)...,
                                 engine);
        },
        std::integer_sequence<bool, false, true, true>{}, out_md, low_mds,
        high_mds);

    return out_md;
}

} // namespace mdtensor::random
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/uniform.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/random/random.hpp
//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/util/util.hpp
/**
 * @file
 * @brief Miscellaneous utility functions for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */


//BEGIN_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/util/broadcast_extents.hpp
/**
 * @file
 * @brief Broadcast extents utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */



namespace mdtensor {

template <core::extents_c... ins_t>
[[nodiscard]] constexpr auto broadcast_extents(ins_t &&...ins) {
    return core::broadcast_extents(std::forward<ins_t>(ins)...);
}

} // namespace mdtensor
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/util/broadcast_extents.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/util/util.hpp
//END_FILE_INCLUDE: /home/runner/work/mdtensor/mdtensor/mdtensor/mdtensor.hpp
#endif // MDTENSOR_SINGLE_HEADER_INCLUDE_GUARD_

