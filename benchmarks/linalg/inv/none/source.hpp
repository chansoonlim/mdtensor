#pragma once

#include "benchmark/benchmark.h"
#include "mdtensor/mdtensor.hpp"

namespace md = mdtensor;

constexpr size_t RANGE_START = 1;
constexpr size_t RANGE_END = 10;
constexpr size_t RANGE_INCREMENT = 1;

template <typename T> inline void test(benchmark::State &state) {
    const size_t set_num = state.range(0);

    const auto in = md::ones<T>(md::dims<2>{set_num, set_num});
    auto out = md::empty<T>(md::dims<2>{set_num, set_num});
    int8_t valid;

    for (auto _ : state) {
        md::linalg::inv_to(in, out, valid);
    }

    state.SetComplexityN(state.range(0));
}
