/**
 * @file
 * @brief add benchmark executable.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

// #include "benchmarks/common/benchmarking.hpp"

// #include "mdtensor/mdtensor.hpp"

// namespace md = mdtensor;

// struct Target {
//     template <typename dtype, md::Backend backend>
//     static inline void run(benchmark::State &state) {
//         const std::size_t len = state.range(0);

//         const auto in1 = std::vector<dtype>(len, 1);
//         const auto in2 = std::vector<dtype>(len, 2);
//         auto out = std::vector<dtype>(len, 0);

//         const dtype *p1 = in1.data();
//         const dtype *p2 = in2.data();
//         dtype *po = out.data();

// #pragma omp target enter data map(to : p1[0 : len], p2[0 : len])               \
//     map(alloc : po[0 : len])

//         for (auto _ : state) {
// #pragma omp target teams distribute parallel for nowait map( \
//         to : p1[0 : len], p2[0 : len]) map(from : po[0 : len])
//             for (std::size_t i = 0; i < len; i++) {
//                 po[i] = p1[i] + p2[i];
//             }
//         }

// #pragma omp target exit data map(delete : p1[0 : len], p2[0 : len], po[0 :
// len])

//         state.SetComplexityN(state.range(0));
//     }
// };

// int main(int argc, char **argv) {
//     benchmarking::Settings settings;

//     settings.benchmark_name = "add";
//     settings.defaults.dtype = "ps";
//     settings.defaults.backend = "auto";
//     settings.defaults.range_end = 10'000'000;

//     return benchmarking::run<Target>(argc, argv, std::move(settings));
// }
