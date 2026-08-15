[English](./README.md)
| [한국어](./README-ko.md)

# mdtensor — A Modern C++ Tensor Library with a NumPy-like Interface

**mdtensor** is a header-only Modern C++ tensor computation library built around non-owning views based on C++23's [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan) and owning containers based on [`kokkos::mdarray`](https://github.com/kokkos/mdspan/blob/stable/include/experimental/mdarray). It aims to provide an interface similar to NumPy while actively leveraging Modern C++ features to maximize compile-time evaluation and optimize runtime performance.

> mdtensor is under active development. Both the internal framework and the public APIs may change substantially. At this stage, please consider it an experimental repository for high-performance tensor computation with `std::mdspan`, rather than a production-ready library.

---

## Why Build Another Numerical Computing Library?

NumPy has evolved around vectorized operations and broadcasting to reduce the overhead of Python-level loops and function calls. Expressing multidimensional computations as array operations moves loops into the library and makes memory access patterns more predictable, making it easier to achieve high performance on modern CPUs. In contrast, function-call overhead is comparatively low in C++, and matrix-oriented libraries such as Eigen have long been widely used in production. As a result, demand for a general-purpose, NumPy-style N-dimensional tensor library has historically been lower.

C++23's [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan) is an important step toward representing multidimensional views in a standardized way in C++. [`std::linalg`](https://en.cppreference.com/w/cpp/numeric/linalg) is also expected to be introduced in C++26. Although the rank of a `std::mdspan` must be known at compile time, it is designed as a zero-overhead abstraction and is therefore well suited to expressing multidimensional computations efficiently in C++.

While researching robotics algorithms, the author has spent many years porting algorithms written in Python to C++ or C to accelerate them on PCs and MCUs. Python provides several acceleration tools, including PyPy and Numba, but performance and deployment constraints often make it necessary to reimplement algorithms in C++ and repeatedly verify that their results match the Python implementation. Applying SIMD and CPU/GPU parallelism then requires additional, separate optimization work.

Research on executing high-performance algorithms on microsecond timescales through SIMD-based computation has also become increasingly active. For example, [Motions in Microseconds via Vectorized Sampling-Based Planning](https://arxiv.org/abs/2309.14545) demonstrates how vectorized computation can improve the performance of robotics algorithms. mdtensor targets precisely this area: **implementing NumPy-like multidimensional broadcasting operations in C++ as efficiently as possible**.

Because `std::mdspan` is relatively new, established libraries may find it difficult to adopt it quickly while preserving backward compatibility. Initial tests also indicate that algorithms built on `std::mdspan` can be optimized aggressively by compilers when combined with loop unrolling. mdtensor is being developed to apply this potential first to robotics software, where real-time computational performance is critical. More mathematical libraries are expected to emerge around the capabilities of `std::mdspan`; accordingly, mdtensor's implementations of fundamental NumPy-equivalent functionality are released under the Apache License 2.0.

mdtensor provides the following features:

- **NumPy-like API**: Familiar APIs for element-wise mathematics, logic, array manipulation, and broadcasting.
- **Views based on [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan)**: Lightweight multidimensional views that refer to external memory without owning it.
- **Containers based on [`kokkos::mdarray`](https://github.com/kokkos/mdspan/blob/stable/include/experimental/mdarray)**: Owning multidimensional containers developed within the `std::mdspan` ecosystem.
- **`constexpr`-oriented implementation**: Designed so that as many operations as possible can be used for compile-time evaluation and testing. Every currently implemented API can be evaluated in a `constexpr` context.
- **Zero-copy broadcasting and manipulation views**: Uses stride-based view transformations rather than eager copies whenever possible.

---

## Quick Example ([Run on Godbolt 😃](https://godbolt.org/z/cYea11TK4))

### 1. Compile-time Addition with Broadcasting

```cpp
#include <iostream>

#include "mdtensor/mdtensor.hpp"

namespace md = mdtensor;

int main() {
    constexpr auto a = md::full<int>(md::extents<std::size_t, 3, 1, 2>{}, 1);
    constexpr auto b = md::full<int>(md::extents<std::size_t, 2, 1>{}, 2);
    constexpr auto c = md::add(a, b);

    constexpr auto c_expect =
        md::full<int>(md::extents<std::size_t, 3, 2, 2>{}, 3);

    constexpr bool is_allclose = md::allclose(c, c_expect);

    std::cout << "a extents: " << md::to_string(a.extents()) << std::endl;
    std::cout << "a: " << md::to_string(a) << std::endl << std::endl;

    std::cout << "b extents: " << md::to_string(b.extents()) << std::endl;
    std::cout << "b: " << md::to_string(b) << std::endl << std::endl;

    std::cout << "c extents: " << md::to_string(c.extents()) << std::endl;
    std::cout << "c: " << md::to_string(c) << std::endl;

    static_assert(is_allclose);
}
```

Result:

```text
a extents: (3, 1, 2)
a: [[[1, 1]], [[1, 1]], [[1, 1]]]

b extents: (2, 1)
b: [[2], [2]]

c extents: (3, 2, 2)
c: [[[3, 3], [3, 3]], [[3, 3], [3, 3]], [[3, 3], [3, 3]]]
```


---

## Currently Implemented API Groups

| Module | mdtensor API | Comparable NumPy API |
|---|---|---|
| Creation | [`empty`](mdtensor/creation/empty.hpp), [`empty_like`](mdtensor/creation/empty_like.hpp), [`eye`](mdtensor/creation/eye.hpp), [`ones`](mdtensor/creation/ones.hpp), [`ones_like`](mdtensor/creation/ones_like.hpp), [`zeros`](mdtensor/creation/zeros.hpp), [`zeros_like`](mdtensor/creation/zeros_like.hpp), [`full`](mdtensor/creation/full.hpp), [`full_like`](mdtensor/creation/full_like.hpp), [`copy`](mdtensor/creation/copy.hpp), [`arange`](mdtensor/creation/arange.hpp), [`linspace`](mdtensor/creation/linspace.hpp) | [`np.empty`](https://numpy.org/doc/stable/reference/generated/numpy.empty.html), [`np.empty_like`](https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html), [`np.eye`](https://numpy.org/doc/stable/reference/generated/numpy.eye.html), [`np.ones`](https://numpy.org/doc/stable/reference/generated/numpy.ones.html), [`np.ones_like`](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html), [`np.zeros`](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html), [`np.zeros_like`](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html), [`np.full`](https://numpy.org/doc/stable/reference/generated/numpy.full.html), [`np.full_like`](https://numpy.org/doc/stable/reference/generated/numpy.full_like.html), [`np.copy`](https://numpy.org/doc/stable/reference/generated/numpy.copy.html), [`np.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html), [`np.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) |
| Manipulation | [`reshape`](mdtensor/manipulation/reshape.hpp), [`flatten`](mdtensor/manipulation/flatten.hpp), [`transpose`](mdtensor/manipulation/transpose.hpp), [`broadcast`](mdtensor/manipulation/broadcast.hpp), [`broadcast_to`](mdtensor/manipulation/broadcast_to.hpp), [`expand_dims`](mdtensor/manipulation/expand_dims.hpp), [`concatenate`](mdtensor/manipulation/concatenate.hpp) | [`np.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html), [`ndarray.flatten`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html), [`np.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html), [`np.broadcast`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html), [`np.broadcast_to`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html), [`np.expand_dims`](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html), [`np.concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html) |
| Logic | [`all`](mdtensor/logic/all.hpp), [`any`](mdtensor/logic/any.hpp), [`isinf`](mdtensor/logic/isinf.hpp), [`isnan`](mdtensor/logic/isnan.hpp), [`logical_and`](mdtensor/logic/logical_and.hpp), [`logical_or`](mdtensor/logic/logical_or.hpp), [`logical_not`](mdtensor/logic/logical_not.hpp), [`logical_xor`](mdtensor/logic/logical_xor.hpp), [`allclose`](mdtensor/logic/allclose.hpp), [`isclose`](mdtensor/logic/isclose.hpp), [`array_equal`](mdtensor/logic/array_equal.hpp), [`array_equiv`](mdtensor/logic/array_equiv.hpp), [`greater`](mdtensor/logic/greater.hpp), [`greater_equal`](mdtensor/logic/greater_equal.hpp), [`less`](mdtensor/logic/less.hpp), [`less_equal`](mdtensor/logic/less_equal.hpp), [`equal`](mdtensor/logic/equal.hpp), [`not_equal`](mdtensor/logic/not_equal.hpp) | [`np.all`](https://numpy.org/doc/stable/reference/generated/numpy.all.html), [`np.any`](https://numpy.org/doc/stable/reference/generated/numpy.any.html), [`np.isinf`](https://numpy.org/doc/stable/reference/generated/numpy.isinf.html), [`np.isnan`](https://numpy.org/doc/stable/reference/generated/numpy.isnan.html), [`np.logical_and`](https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html), [`np.logical_or`](https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html), [`np.logical_not`](https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html), [`np.logical_xor`](https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html), [`np.allclose`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html), [`np.isclose`](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html), [`np.array_equal`](https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html), [`np.array_equiv`](https://numpy.org/doc/stable/reference/generated/numpy.array_equiv.html), [`np.greater`](https://numpy.org/doc/stable/reference/generated/numpy.greater.html), [`np.greater_equal`](https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html), [`np.less`](https://numpy.org/doc/stable/reference/generated/numpy.less.html), [`np.less_equal`](https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html), [`np.equal`](https://numpy.org/doc/stable/reference/generated/numpy.equal.html), [`np.not_equal`](https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html) |
| Math | [`sin`](mdtensor/math/sin.hpp), [`cos`](mdtensor/math/cos.hpp), [`tan`](mdtensor/math/tan.hpp), [`atan2`](mdtensor/math/atan2.hpp), [`deg2rad`](mdtensor/math/deg2rad.hpp), [`rad2deg`](mdtensor/math/rad2deg.hpp), [`sum`](mdtensor/math/sum.hpp), [`add`](mdtensor/math/add.hpp), [`negative`](mdtensor/math/negative.hpp), [`multiply`](mdtensor/math/multiply.hpp), [`divide`](mdtensor/math/divide.hpp), [`subtract`](mdtensor/math/subtract.hpp), [`maximum`](mdtensor/math/maximum.hpp), [`max`](mdtensor/math/max.hpp), [`nanmax`](mdtensor/math/nanmax.hpp), [`minimum`](mdtensor/math/minimum.hpp), [`min`](mdtensor/math/min.hpp), [`nanmin`](mdtensor/math/nanmin.hpp), [`clip`](mdtensor/math/clip.hpp), [`sqrt`](mdtensor/math/sqrt.hpp), [`absolute`](mdtensor/math/absolute.hpp), [`sign`](mdtensor/math/sign.hpp) | [`np.sin`](https://numpy.org/doc/stable/reference/generated/numpy.sin.html), [`np.cos`](https://numpy.org/doc/stable/reference/generated/numpy.cos.html), [`np.tan`](https://numpy.org/doc/stable/reference/generated/numpy.tan.html), [`np.atan2`](https://numpy.org/doc/stable/reference/generated/numpy.atan2.html), [`np.deg2rad`](https://numpy.org/doc/stable/reference/generated/numpy.deg2rad.html), [`np.rad2deg`](https://numpy.org/doc/stable/reference/generated/numpy.rad2deg.html), [`np.sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html), [`np.add`](https://numpy.org/doc/stable/reference/generated/numpy.add.html), [`np.negative`](https://numpy.org/doc/stable/reference/generated/numpy.negative.html), [`np.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html), [`np.divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html), [`np.subtract`](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html), [`np.maximum`](https://numpy.org/doc/stable/reference/generated/numpy.maximum.html), [`np.max`](https://numpy.org/doc/stable/reference/generated/numpy.max.html), [`np.nanmax`](https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html), [`np.minimum`](https://numpy.org/doc/stable/reference/generated/numpy.minimum.html), [`np.min`](https://numpy.org/doc/stable/reference/generated/numpy.min.html), [`np.nanmin`](https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html), [`np.clip`](https://numpy.org/doc/stable/reference/generated/numpy.clip.html), [`np.sqrt`](https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html), [`np.absolute`](https://numpy.org/doc/stable/reference/generated/numpy.absolute.html), [`np.sign`](https://numpy.org/doc/stable/reference/generated/numpy.sign.html) |
| Linear algebra | [`matmul`](mdtensor/linalg/matmul.hpp), [`matvec`](mdtensor/linalg/matvec.hpp), [`vecmat`](mdtensor/linalg/vecmat.hpp), [`cholesky`](mdtensor/linalg/cholesky.hpp), [`norm`](mdtensor/linalg/norm.hpp), [`inv`](mdtensor/linalg/inv.hpp), [`lu`](mdtensor/linalg/lu.hpp), [`solve`](mdtensor/linalg/solve.hpp) | [`np.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html), [`np.matvec`](https://numpy.org/doc/stable/reference/generated/numpy.matvec.html), [`np.vecmat`](https://numpy.org/doc/stable/reference/generated/numpy.vecmat.html), [`np.linalg.cholesky`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html), [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html), [`np.linalg.inv`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html), [`scipy.linalg.lu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu.html), [`np.linalg.solve`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html) |
| Random | [`rand`](mdtensor/random/rand.hpp), [`randint`](mdtensor/random/randint.hpp), [`uniform`](mdtensor/random/uniform.hpp) | [`np.random.rand`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html), [`np.random.randint`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html), [`np.random.uniform`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html) |
| Utility | [`broadcast_extents`](mdtensor/util/broadcast_extents.hpp), [`fill`](mdtensor/util/fill.hpp) | [`np.broadcast_shapes`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_shapes.html), [`ndarray.fill`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.fill.html) |

---

## Installation

mdtensor is a header-only library. Add the repository to your compiler's include path and include the following header. A compiler with C++20 or later support is required.

```cpp
#include "mdtensor/mdtensor.hpp"
```

---

## Tests and Benchmarks

This repository uses the [Bazel](https://bazel.build/) build system. After installing Bazel, you can run the tests and benchmarks with the commands below. The development environment defined by `.devcontainer/dockerfile` can run them without additional setup.

### 1. Run All Tests

```bash
bazel test tests/...
```

### 2. Run a Benchmark

```bash
bazel run benchmarks/math/add:main
```

---

## Roadmap

- Expand NumPy API coverage
- Strengthen generalized broadcasting rules
- Introduce lazy computation
- Select the optimal backend based on compile-time and runtime workload analysis, including SIMD, Eigen, and CPU/GPU multiprocessor backends

---

## Contributing

This project is actively maintained, although new APIs and framework changes are currently driven primarily by the author's needs. Feel free to open an issue or pull request if you are interested in developing new APIs, improving the framework, or extending backend support.

---

## License

mdtensor is distributed under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.

mdtensor contains some code derived from CTMD v0.16.1, released on August 19, 2025. CTMD is an Apache License 2.0 library developed by the author of this project at Uon Robotics. Since then, mdtensor has been substantially modified through API extensions and framework improvements.

The original copyright for CTMD is held by Uon Robotics. The modifications and extensions in mdtensor are maintained by Chan-Soon Lim.
