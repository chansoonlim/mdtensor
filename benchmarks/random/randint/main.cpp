/**
 * @file
 * @brief benchmark executable.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#define BENCHMARK_SKIP_FLOATING_POINT_TYPES

#include "benchmarks/common/benchmarking.hpp"

#include "mdtensor/mdtensor.hpp"

namespace md = mdtensor;

struct Target {
    template <typename dtype, md::Backend backend>
    static inline void run(benchmark::State &state) {
        const auto len = static_cast<std::size_t>(state.range(0));

        auto out = md::empty<dtype>(len);

        benchmark::ClobberMemory();

        for (auto _ : state) {
            static_cast<void>(md::random::randint<void>(
                out.extents(), std::nullopt, std::nullopt, out));
            benchmark::ClobberMemory();
        }

        state.SetComplexityN(state.range(0));
    }
};

int main(int argc, char **argv) {
    benchmarking::Settings settings;

    settings.benchmark_name = "randint";
    settings.defaults.dtype = "i32";
    settings.defaults.backend = "all";
    settings.defaults.range_end = 10'000'000;

    return benchmarking::run<Target>(argc, argv, std::move(settings));
}
