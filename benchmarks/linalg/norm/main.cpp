/**
 * @file
 * @brief benchmark executable.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#define BENCHMARK_SKIP_INTEGER_TYPES

#define BENCHMARK_MPMODE_SIMD

#ifdef MDTENSOR_USE_EIGEN
#define BENCHMARK_MPMODE_EIGEN
#endif

#include "benchmarks/common/benchmarking.hpp"

#include "mdtensor/mdtensor.hpp"

namespace md = mdtensor;

struct Target {
    template <typename dtype, md::Backend backend>
    static inline void run(benchmark::State &state) {
        const auto len = static_cast<std::size_t>(state.range(0));

        const auto in = md::ones<dtype>(md::dims<1>{len});

        benchmark::ClobberMemory();

        for (auto _ : state) {
            static_cast<void>(md::linalg::norm<void, backend>(in));
            benchmark::ClobberMemory();
        }

        state.SetComplexityN(state.range(0));
    }
};

int main(int argc, char **argv) {
    benchmarking::Settings settings;

    settings.benchmark_name = "norm";
    settings.defaults.dtype = "ps";
    settings.defaults.backend = "all";

    settings.defaults.range_multiplier = false;
    settings.defaults.range_step = 10;
    settings.defaults.range_end = 100;

    return benchmarking::run<Target>(argc, argv, std::move(settings));
}
