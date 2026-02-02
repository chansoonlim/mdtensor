#pragma once

#include "benchmark/benchmark.h"
#include "mdtensor/mdtensor.hpp"

namespace md = mdtensor;

constexpr size_t RANGE_START = 1;
constexpr size_t RANGE_END = 10;
constexpr size_t RANGE_INCREMENT = 1;

template <typename T> inline void test(benchmark::State &state) {
    const size_t set_num = state.range(0);

    const auto in1 = md::ones<T>(md::dims<2>{set_num, set_num});
    const auto in2 = md::ones<T>(md::dims<1>{set_num});
    auto out = md::empty<T>(md::dims<1>{set_num});

    for (auto _ : state) {
        md::linalg::matvec_to(in1, in2, out);
    }

    state.SetComplexityN(state.range(0));
}
