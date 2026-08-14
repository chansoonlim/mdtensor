/**
 * @file
 * @brief add benchmark executable.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#include "benchmarks/common/benchmarking.hpp"

#include "mdtensor/mdtensor.hpp"

namespace md = mdtensor;

struct Target {
    template <typename dtype, md::Backend backend>
    static inline void run(benchmark::State &state) {
        const auto len = static_cast<std::size_t>(state.range(0));

        const auto in1 = std::vector<dtype>(len, 1);
        const auto in2 = std::vector<dtype>(len, 2);
        auto out = std::vector<dtype>(len, 0);

        benchmark::ClobberMemory();

        for (auto _ : state) {
            for (std::size_t i = 0; i < len; i++) {
                out[i] = in1[i] + in2[i];
            }
            benchmark::ClobberMemory();
        }

        state.SetComplexityN(state.range(0));
    }
};

int main(int argc, char **argv) {
    benchmarking::Settings settings;

    settings.benchmark_name = "add";
    settings.defaults.dtype = "ps";
    settings.defaults.backend = "auto";
    settings.defaults.range_end = 10'000'000;

    return benchmarking::run<Target>(argc, argv, std::move(settings));
}
