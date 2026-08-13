/**
 * @file
 * @brief benchmark executable.
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

        const auto in = [&]() {
            if constexpr (std::is_floating_point_v<dtype>) {
                return md::random::rand<dtype>(md::dims<2>{len, len});

            } else {
                return md::random::randint<dtype>(md::dims<2>{len, len});
            }
        }();

        auto out = md::empty<dtype>(md::dims<1>{len});

        benchmark::ClobberMemory();

        for (auto _ : state) {
            static_cast<void>(
                md::any<false, backend>(in, std::index_sequence<0>{}, out));
            benchmark::ClobberMemory();
        }

        state.SetComplexityN(state.range(0));
    }
};

int main(int argc, char **argv) {
    benchmarking::Settings settings;

    settings.benchmark_name = "any";
    settings.defaults.dtype = "b";
    settings.defaults.backend = "auto";
    settings.defaults.range_end = 10'000;

    return benchmarking::run<Target>(argc, argv, std::move(settings));
}
