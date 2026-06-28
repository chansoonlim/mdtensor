# mdtensor — Modern C++ Tensor Library with Numpy-like Syntax

## Overview

**mdtensor** is a header-only C++ library providing multi-dimensional tensor operations with NumPy-like broadcasting semantics.

It is built on modern C++ standard facilities including:
- C++23 [std::mdspan](https://en.cppreference.com/w/cpp/container/mdspan).
- Upcoming C++26 [std::mdarray](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p1684r2.html) and [std::submdspan](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2630r4.html).

Try it on Godbolt:
  - [Matrix Addition example](https://godbolt.org/z/vvEE3crfq).
  - [Zero-cost Abstraction test](https://godbolt.org/z/5qYPYo9Pa).

## Key Features

- **NumPy-like Syntax**: Familiar tensor programming model inspired by NumPy.
- **Flexibility**: Supports diverse data types compatible with [std::mdspan](https://en.cppreference.com/w/cpp/container/mdspan).
- **Zero-Cost Abstraction**: Broadcasting and tensor operations are implemented with minimal runtime overhead.
- **Compile-Time Computation Support**: Most operations are usable in [constexpr](https://en.cppreference.com/w/cpp/language/constexpr.html) contexts.
- **Optional High-Performance Backends**: [Eigen](https://libeigen.gitlab.io/) and [OpenMP](https://www.openmp.org/) acceleration available when enabled.
- **Header-only & Lightweight**: No build system required — just include the headers.

## Important Notes

- C++ Version Compatibility:
  - mdtensor requires at least C++20 for full functionality.
  - [std::mdspan](https://en.cppreference.com/w/cpp/container/mdspan) (C++23), [std::mdarray](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p1684r2.html) (C++26), [std::submdspan](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2630r4.html) (C++26) are sourced from the [Reference mdspan implementation](https://github.com/kokkos/mdspan).

- Data conventions:
  - Arguments:
    - **Scalar**: 0D mdspan or arithmetic type (e.g., int, float, double).
    - **Vector**: 1D mdspan or 1D mdarray.
    - **Matrix**: 2D mdspan or 2D mdarray.
    - **Tensor**: ND mdspan or ND mdarray.

  - Return Values:
    - **Scalar**: arithmetic type (e.g., int, float, double).
    - **Vector**: 1D mdarray.
    - **Matrix**: 2D mdarray.
    - **Tensor**: ND mdarray.

- Acceleration:
  - Using [Eigen](https://libeigen.gitlab.io/): define the macro `MDTENSOR_USE_EIGEN`.
  - Using [OpenMP](https://www.openmp.org/): compile with `-fopenmp` and link with `-lgomp`.

## Available Functions

- **Array creation routines**: [empty](mdtensor/creation/empty.hpp), [empty_like](mdtensor/creation/empty_like.hpp), [eye](mdtensor/creation/eye.hpp), [ones](mdtensor/creation/ones.hpp), [ones_like](mdtensor/creation/ones_like.hpp), [zeros](mdtensor/creation/zeros.hpp), [zeros_like](mdtensor/creation/zeros_like.hpp), [full](mdtensor/creation/full.hpp), [full_like](mdtensor/creation/full_like.hpp), [copy](mdtensor/creation/copy.hpp), [arange](mdtensor/creation/arange.hpp), [linspace](mdtensor/creation/linspace.hpp).
- **Array manipulation routines**: [reshape](mdtensor/manipulation/reshape.hpp), [expand_dims](mdtensor/manipulation/expand_dims.hpp), [concatenate](mdtensor/manipulation/concatenate.hpp).
- **Linear algebra**: [matmul](mdtensor/linalg/matmul.hpp), [matvec](mdtensor/linalg/matvec.hpp), [vecmat](mdtensor/linalg/vecmat.hpp), [norm](mdtensor/linalg/norm.hpp), [inv](mdtensor/linalg/inv.hpp).
- **Logic functions**: [all](mdtensor/logic/all.hpp), [any](mdtensor/logic/any.hpp), [allclose](mdtensor/logic/allclose.hpp), [isclose](mdtensor/logic/isclose.hpp), [array_equal](mdtensor/logic/array_equal.hpp), [array_equiv](mdtensor/logic/array_equiv.hpp), [greater](mdtensor/logic/greater.hpp), [greater_equal](mdtensor/logic/greater_equal.hpp), [less](mdtensor/logic/less.hpp), [less_equal](mdtensor/logic/less_equal.hpp), [equal](mdtensor/logic/equal.hpp), [not_equal](mdtensor/logic/not_equal.hpp).
- **Mathematical functions**: [sin](mdtensor/math/sin.hpp), [cos](mdtensor/math/cos.hpp), [tan](mdtensor/math/tan.hpp), [atan2](mdtensor/math/atan2.hpp), [deg2rad](mdtensor/math/deg2rad.hpp), [rad2deg](mdtensor/math/rad2deg.hpp), [sum](mdtensor/math/sum.hpp), [add](mdtensor/math/add.hpp), [negative](mdtensor/math/negative.hpp), [multiply](mdtensor/math/multiply.hpp), [divide](mdtensor/math/divide.hpp), [subtract](mdtensor/math/subtract.hpp), [maximum](mdtensor/math/maximum.hpp), [max](mdtensor/math/max.hpp), [minimum](mdtensor/math/minimum.hpp), [min](mdtensor/math/min.hpp), [clip](mdtensor/math/clip.hpp), [sqrt](mdtensor/math/sqrt.hpp), [absolute](mdtensor/math/absolute.hpp), [sign](mdtensor/math/sign.hpp).
- **Random sampling**: [rand](mdtensor/random/rand.hpp), [uniform](mdtensor/random/uniform.hpp).


## Installation
mdtensor is a header-only library, so you can start using it by simply including:
  ```cpp
  #include "mdtensor/mdtensor.hpp"
  ```


## Examples

### Matrix Addition with Broadcasting

  Code:
  ```cpp
  #include "mdtensor/mdtensor.hpp"

  namespace md = mdtensor;

  constexpr auto a = md::full<int>(1, md::extents<size_t, 3, 1, 2>{});
  constexpr auto b = md::full<int>(2, md::extents<size_t, 2, 1>{});
  constexpr auto c = md::add(a, b);

  constexpr auto c_expect = md::full<int>(3, md::extents<size_t, 3, 2, 2>{});

  constexpr bool is_allclose = md::allclose(c, c_expect);

  std::cout << "a extents: " << md::to_string(a.extents()) << std::endl;
  std::cout << "a: " << md::to_string(a) << std::endl << std::endl;

  std::cout << "b extents: " << md::to_string(b.extents()) << std::endl;
  std::cout << "b: " << md::to_string(b) << std::endl << std::endl;

  std::cout << "c extents: " << md::to_string(c.extents()) << std::endl;
  std::cout << "c: " << md::to_string(c) << std::endl;

  static_assert(is_allclose);
  ```

  Output:
  ```bash
  a extents: (3, 1, 2)
  a: [[[1, 1]], [[1, 1]], [[1, 1]]]

  b extents: (2, 1)
  b: [[2], [2]]

  c extents: (3, 2, 2)
  c: [[[3, 3], [3, 3]], [[3, 3], [3, 3]], [[3, 3], [3, 3]]]
  ```

## Tests and Benchmarks
mdtensor uses [GoogleTest](https://github.com/google/googletest) for unit tests:
  ```bash
  bazel test tests/...
  ```

Benchmarks use [GoogleBenchmark](https://github.com/google/benchmark). For example:
  ```bash
  bazel run benchmarks/add/none:ps
  ```

## About
This project includes derivative works based on:
- [CTMD (Compile-Time Multi-Dimensional)](https://github.com/uonrobotics/ctmd/tree/v0.16.1) library v0.16.1
- Copyright (c) 2025 Uon Robotics, South Korea
- Licensed under the Apache License 2.0.

CTMD was originally developed at Uon Robotics by Chan-Soon Lim, and later modified and extended as part of the mdtensor project.
All original rights to CTMD remain with Uon Robotics, while modifications and extensions in mdtensor are maintained by Chan-Soon Lim.


## Contributing
mdtensor is in active development, and contributions are welcome!
Feel free to open issues or pull requests — whether for bug reports, feature ideas, or improvements.

## License
mdtensor is licensed under the Apache 2.0 License.
See the [LICENSE](LICENSE) file for full details.
