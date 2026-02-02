#pragma once

#include "benchmark/benchmark.h"
#include "mdtensor/mdtensor.hpp"

namespace md = mdtensor;

constexpr size_t RANGE_START = 1;
constexpr size_t RANGE_END = 10;
constexpr size_t RANGE_INCREMENT = 1;

template <typename T> inline void test(benchmark::State &state) {
    const size_t set_num = state.range(0);

    const auto in = md::ones<T>(md::dims<1>{set_num});
    T out;

    for (auto _ : state) {
        md::linalg::norm_to(in, out);
    }

    state.SetComplexityN(state.range(0));
}
