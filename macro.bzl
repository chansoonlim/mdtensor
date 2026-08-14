load("@rules_cc//cc:defs.bzl", "cc_test", "cc_binary")

def mdtensor_cc_test(name = "main", srcs = ["main.cpp"], deps = [], copts = []):
    cc_test(
        name = name,
        srcs = srcs,
        copts =  [
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wpedantic",
        ] + copts,
        deps = [
            "@googletest//:gtest_main",
            "//:mdtensor",
        ] + deps,
    )

def mdtensor_benchmark(name = "main", srcs = ["main.cpp"], deps = [], copts = []):
    cc_binary(
        name = name,
        srcs = srcs,
        copts =  [
            "-O3",
            "-funroll-loops",
            "-march=native",
            "-fopenmp",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wpedantic",
        ] + copts,
        linkopts = [
            "-fopenmp",
        ],
        deps = [
            "//benchmarks/common:benchmarking",
        ] + deps,
    )
