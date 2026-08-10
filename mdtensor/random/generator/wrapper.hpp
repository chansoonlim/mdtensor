/**
 * @file
 * @brief Random number generator engine wrapper for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

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
