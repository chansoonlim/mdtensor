/**
 * @file
 * @brief Common runtime dispatcher for mdtensor benchmarks.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include <benchmark/benchmark.h>

#include "mdtensor/mdtensor.hpp"

namespace md = mdtensor;

namespace benchmarking {

struct Options {
    std::string dtype = "ps";
    std::string backend = "none";
    bool range_multiplier = true;
    std::size_t range_start = 1;
    std::size_t range_end = 1'000'000;
    std::size_t range_step = 10;
};

struct Settings {
    Options defaults{};
    std::string benchmark_name{};
    std::string context_prefix = "mdtensor";
    benchmark::TimeUnit unit = benchmark::kMicrosecond;
    benchmark::BigO complexity = benchmark::oN;
    bool use_real_time = true;
};

namespace detail {

[[nodiscard]] inline std::optional<std::string_view>
option_value(std::string_view argument, std::string_view option_name,
             int &read_index, int argc, char **argv) {
    if (argument == option_name) {
        if (++read_index >= argc) {
            throw std::invalid_argument(std::string(option_name) +
                                        " requires a value");
        }
        return std::string_view(argv[read_index]);
    }

    if (argument.size() > option_name.size() &&
        argument.compare(0, option_name.size(), option_name) == 0 &&
        argument[option_name.size()] == '=') {
        return argument.substr(option_name.size() + 1);
    }

    return std::nullopt;
}

[[nodiscard]] inline std::size_t parse_integer(std::string_view text,
                                               std::string_view option_name) {
    std::size_t value{};
    const auto [ptr, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);

    if (error != std::errc{} || ptr != text.data() + text.size()) {
        throw std::invalid_argument(
            std::string(option_name) +
            " has an invalid integer value: " + std::string(text));
    }

    return value;
}

[[nodiscard]] inline Options parse_options(int &argc, char **argv,
                                           Options options = {}) {
    int write_index = 1;

    for (int read_index = 1; read_index < argc; ++read_index) {
        const std::string_view argument(argv[read_index]);

        if (const auto value =
                option_value(argument, "--dtype", read_index, argc, argv)) {
            options.dtype = *value;

        } else if (const auto value = option_value(argument, "--backend",
                                                   read_index, argc, argv)) {
            options.backend = *value;

        } else if (const auto value = option_value(argument, "--range_start",
                                                   read_index, argc, argv)) {
            options.range_start = parse_integer(*value, "--range_start");

        } else if (const auto value = option_value(argument, "--range_end",
                                                   read_index, argc, argv)) {
            options.range_end = parse_integer(*value, "--range_end");

        } else if (const auto value = option_value(argument, "--range_step",
                                                   read_index, argc, argv)) {
            options.range_step = parse_integer(*value, "--range_step");

        } else {
            argv[write_index++] = argv[read_index];
        }
    }

    argc = write_index;
    argv[write_index] = nullptr;

    return options;
}

template <typename Target, typename dtype, md::Backend backend>
[[nodiscard]] consteval bool target_supports() {
    if constexpr (requires {
                      {
                          Target::template supports<dtype, backend>()
                      } -> std::convertible_to<bool>;
                  }) {
        return static_cast<bool>(Target::template supports<dtype, backend>());
    } else {
        return true;
    }
}

[[nodiscard]] inline std::string
make_benchmark_name(const Settings &settings, std::string_view dtype_name,
                    std::string_view mpmode_name) {
    const std::string prefix = settings.benchmark_name.empty()
                                   ? std::string{}
                                   : settings.benchmark_name + "/";
    return prefix + std::string(dtype_name) + "/" + std::string(mpmode_name);
}

template <typename Target, typename dtype, md::Backend backend>
[[nodiscard]] inline std::size_t
register_one(const Options &options, const Settings &settings,
             std::string_view dtype_name, std::string_view mpmode_name) {
    if constexpr (!target_supports<Target, dtype, backend>()) {
        return 0;
    } else {
        const std::string name =
            make_benchmark_name(settings, dtype_name, mpmode_name);

        auto *registered = benchmark::RegisterBenchmark(
            name.c_str(), [](benchmark::State &state) {
                Target::template run<dtype, backend>(state);
            });

        registered->Unit(settings.unit);

        if (options.range_multiplier) {
            registered->RangeMultiplier(options.range_step)
                ->Range(options.range_start, options.range_end);

        } else {
            registered->DenseRange(options.range_start, options.range_end,
                                   options.range_step);
        }

        if (settings.use_real_time) {
            registered->UseRealTime();
        }
        if (settings.complexity != benchmark::oNone) {
            registered->Complexity(settings.complexity);
        }

        return 1;
    }
}

template <typename Target, typename dtype>
[[nodiscard]] inline std::size_t register_modes(const Options &options,
                                                const Settings &settings,
                                                std::string_view dtype_name) {
    const bool all = options.backend == "all";
    std::size_t count = 0;

    const auto register_mode = [&]<md::Backend backend>(std::string_view name) {
        if (all || options.backend == name) {
            count += register_one<Target, dtype, backend>(options, settings,
                                                          dtype_name, name);
        }
    };

    register_mode.template operator()<md::Backend::AUTO>("auto");
    register_mode.template operator()<md::Backend::NATIVE>("native");

#ifdef BENCHMARK_MPMODE_SIMD
    register_mode.template operator()<md::Backend::SIMD>("simd");
#endif

#ifdef MDTENSOR_USE_EIGEN
    register_mode.template operator()<md::Backend::EIGEN>("eigen");
#endif

#ifdef MDTENSOR_USE_OPENMP
    register_mode.template operator()<md::Backend::OPENMP>("openmp");
#endif

    return count;
}

template <typename Target>
[[nodiscard]] inline std::size_t register_benchmarks(const Options &options,
                                                     const Settings &settings) {
    const bool all = options.dtype == "all";
    std::size_t count = 0;

    const auto register_dtype = [&]<typename dtype>(std::string_view name) {
        if (all || options.dtype == name) {
            count += register_modes<Target, dtype>(options, settings, name);
        }
    };

#ifndef BENCHMARK_SKIP_INTEGER_TYPES
    register_dtype.template operator()<bool>("b");
    register_dtype.template operator()<std::int8_t>("i8");
    register_dtype.template operator()<std::uint8_t>("u8");
    register_dtype.template operator()<std::int16_t>("i16");
    register_dtype.template operator()<std::uint16_t>("u16");
    register_dtype.template operator()<std::int32_t>("i32");
    register_dtype.template operator()<std::uint32_t>("u32");
    register_dtype.template operator()<std::int64_t>("i64");
    register_dtype.template operator()<std::uint64_t>("u64");
#endif

#ifndef BENCHMARK_SKIP_FLOATING_POINT_TYPES
    register_dtype.template operator()<float>("ps");
    register_dtype.template operator()<double>("pd");
    register_dtype.template operator()<long double>("pl");
#endif

    return count;
}

inline void add_context(const Options &options, const Settings &settings) {
    if (settings.context_prefix.empty()) {
        return;
    }

    benchmark::AddCustomContext(settings.context_prefix + ".dtype",
                                options.dtype);
    benchmark::AddCustomContext(settings.context_prefix + ".backend",
                                options.backend);
}

} // namespace detail

template <typename Target>
[[nodiscard]] inline int run(int argc, char **argv, Settings settings = {}) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);

    try {
        const Options options =
            detail::parse_options(argc, argv, std::move(settings.defaults));

        const std::size_t registered_count =
            detail::register_benchmarks<Target>(options, settings);
        if (registered_count == 0) {
            std::cerr << "No benchmark combination matched dtype="
                      << options.dtype << ", backend=" << options.backend
                      << '\n';
            return EXIT_FAILURE;
        }

        detail::add_context(options, settings);

        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
            benchmark::Shutdown();
            return EXIT_FAILURE;
        }

        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        return EXIT_SUCCESS;

    } catch (const std::exception &error) {
        std::cerr << "Benchmark option error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

} // namespace benchmarking
