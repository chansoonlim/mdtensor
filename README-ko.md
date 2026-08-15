[한국어](./README-ko.md)
| [English](./README.md)

# mdtensor — NumPy-like 문법을 지향하는 Modern C++ Tensor Library

**mdtensor**는 [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan)(C++23) 기반 view와 [`kokkos::mdarray`](https://github.com/kokkos/mdspan/blob/stable/include/experimental/mdarray) 기반 container를 중심으로 설계된 header-only Modern C++ 텐서 연산  라이브러리입니다. 목표는 Numpy와 가까운 인터페이스를 구현하되 Modern C++의 기능을 적극 활용하여 compile-time 연산 극대화, run-time 연산 최적화를 이루는 것입니다.

> 현재 mdtensor는 활발하게 개발 중에 있습니다. 이에 내부 프레임워크를 포함해 외부 API까지 많은 것들이 다양한 방향에서 변경될 수 있습니다. Production level보다는 std::mdspan을 사용한 고속화 테스트 repository로 판단 부탁드립니다.

---

## 왜 또다시 연산 라이브러리를 만드는가?

NumPy는 Python 레벨의 반복문과 함수 호출 오버헤드를 줄이기 위해 vectorized operation과 broadcasting을 적극적으로 활용하도록 발전해 왔습니다. 다차원 연산을 하나의 배열 연산으로 표현하면 반복문을 라이브러리 내부로 밀어 넣을 수 있고, 메모리 접근 패턴도 더 예측 가능해져 현대 CPU에서 높은 성능을 얻기 쉽습니다. 반면 C++에서는 함수 호출 오버헤드가 상대적으로 작고, 실무에서는 Eigen과 같은 행렬 중심 라이브러리가 널리 사용되어 왔기 때문에 NumPy 스타일의 일반적인 N차원 텐서 연산 라이브러리에 대한 수요가 비교적 작았습니다.

C++23에 추가된 [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan)은 C++에서도 다차원 view를 표준적인 방식으로 표현하려는 중요한 시도입니다. 또한 [`std::linalg`](https://en.cppreference.com/w/cpp/numeric/linalg)는 C++26에 추가될 예정입니다. `std::mdspan`은 rank가 compile-time에 결정되어야 한다는 제약이 있지만, zero-overhead abstraction을 목표로 설계되어 C++에서 다차원 연산을 효율적으로 표현하기에 적합합니다.

저자는 로보틱스 알고리즘을 연구하며 Python 기반 알고리즘 코드를 C++ 또는 C로 옮겨 PC와 MCU 환경에서 고속화하는 작업을 오랫동안 수행해 왔습니다. Python에도 PyPy, Numba 등 다양한 고속화 도구가 있지만, 성능과 배포 환경의 제약 때문에 결국 C++에서 알고리즘을 다시 구현하고 Python 결과와 일치시키는 과정이 반복되는 경우가 많았습니다. 여기에 SIMD, CPU/GPU 병렬화까지 적용하려면 매번 별도의 최적화 작업이 필요했습니다.

최근에는 SIMD 기반 계산을 통해 고성능 알고리즘을 microseconds 단위에서 수행하려는 연구도 활발히 진행되고 있습니다. 예를 들어 [Motions in Microseconds via Vectorized Sampling-Based Planning](https://arxiv.org/abs/2309.14545)은 vectorized computation이 로보틱스 알고리즘 성능에 미치는 가능성을 잘 보여줍니다. mdtensor는 바로 이 영역, "**C++에서 NumPy와 같은 다차원 broadcasting 연산을 최대 속도로 구현하는 것**"을 목표로 합니다.

std::mdspan은 매우 최근에 추가된 기능으로 기존 라이브러리들은 이전 버전과의 호환성을 유지하기 위해 빠른 도입은 어려울것이라 판단하고 있습니다. 또한 일부 테스트를 통해 std::mdspan을 활용한 알고리즘이 loop unrolling과 연동되어 컴파일러가 적극적으로 고속화하기 용이함을 확인했습니다. 이에 실시간 계산 속도가 중요한 로봇 프로그램에 선행 적용하기 위해 mdtensor을 신규 개발하고 있으며, 또한 mdspan의 강력한 기능을 활용한 수학 연산 라이브러리는 계속해서 나올것이라 판단하기에 numpy에 존재하는 기본 기능에 대해서 Apache License 2.0을 통해 공개합니다.

mdtensor는 다음을 제공합니다.

- **NumPy-like API**: NumPy에서 익숙한 element-wise math, logic, manipulation, broadcasting 스타일의 API.
- **[`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan) 기반 view**: 외부 메모리를 소유하지 않고 참조할 수 있는 경량 다차원 view.
- **[`kokkos::mdarray`](https://github.com/kokkos/mdspan/blob/stable/include/experimental/mdarray) 기반 container**: std::mdspan 생태계에서 함께 개발된 owning multidimensional container.
- **`constexpr` 지향 구현**: 가능한 많은 연산을 compile-time 테스트 또는 계산에 사용할 수 있도록 설계. 현재 구현된 API는 모두 constexpr로 실행 가능.
- **Zero-copy broadcasting/manipulation view**: 가능한 경우 eager copy 대신 stride 기반 view 변환 사용.

---

## 빠른 예제 ([링크를 누르면 Godbolt에서 실행 가능합니다 😃](https://godbolt.org/z/cYea11TK4))

### 1. Compile-time addition with broadcast
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
```bash
a extents: (3, 1, 2)
a: [[[1, 1]], [[1, 1]], [[1, 1]]]

b extents: (2, 1)
b: [[2], [2]]

c extents: (3, 2, 2)
c: [[[3, 3], [3, 3]], [[3, 3], [3, 3]], [[3, 3], [3, 3]]]
```

---

## 기 개발된 API 그룹

| Module | Mdtensor API | 유사 NumPy 함수 |
|---|---|---|
| Creation | [`empty`](mdtensor/creation/empty.hpp), [`empty_like`](mdtensor/creation/empty_like.hpp), [`eye`](mdtensor/creation/eye.hpp), [`ones`](mdtensor/creation/ones.hpp), [`ones_like`](mdtensor/creation/ones_like.hpp), [`zeros`](mdtensor/creation/zeros.hpp), [`zeros_like`](mdtensor/creation/zeros_like.hpp), [`full`](mdtensor/creation/full.hpp), [`full_like`](mdtensor/creation/full_like.hpp), [`copy`](mdtensor/creation/copy.hpp), [`arange`](mdtensor/creation/arange.hpp), [`linspace`](mdtensor/creation/linspace.hpp) | [`np.empty`](https://numpy.org/doc/stable/reference/generated/numpy.empty.html), [`np.empty_like`](https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html), [`np.eye`](https://numpy.org/doc/stable/reference/generated/numpy.eye.html), [`np.ones`](https://numpy.org/doc/stable/reference/generated/numpy.ones.html), [`np.ones_like`](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html), [`np.zeros`](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html), [`np.zeros_like`](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html), [`np.full`](https://numpy.org/doc/stable/reference/generated/numpy.full.html), [`np.full_like`](https://numpy.org/doc/stable/reference/generated/numpy.full_like.html), [`np.copy`](https://numpy.org/doc/stable/reference/generated/numpy.copy.html), [`np.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html), [`np.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) |
| Manipulation | [`reshape`](mdtensor/manipulation/reshape.hpp), [`flatten`](mdtensor/manipulation/flatten.hpp), [`transpose`](mdtensor/manipulation/transpose.hpp), [`broadcast`](mdtensor/manipulation/broadcast.hpp), [`broadcast_to`](mdtensor/manipulation/broadcast_to.hpp), [`expand_dims`](mdtensor/manipulation/expand_dims.hpp), [`concatenate`](mdtensor/manipulation/concatenate.hpp) | [`np.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html), [`ndarray.flatten`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html), [`np.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html), [`np.broadcast`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html), [`np.broadcast_to`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html), [`np.expand_dims`](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html), [`np.concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html) |
| Logic | [`all`](mdtensor/logic/all.hpp), [`any`](mdtensor/logic/any.hpp), [`isinf`](mdtensor/logic/isinf.hpp), [`isnan`](mdtensor/logic/isnan.hpp), [`logical_and`](mdtensor/logic/logical_and.hpp), [`logical_or`](mdtensor/logic/logical_or.hpp), [`logical_not`](mdtensor/logic/logical_not.hpp), [`logical_xor`](mdtensor/logic/logical_xor.hpp), [`allclose`](mdtensor/logic/allclose.hpp), [`isclose`](mdtensor/logic/isclose.hpp), [`array_equal`](mdtensor/logic/array_equal.hpp), [`array_equiv`](mdtensor/logic/array_equiv.hpp), [`greater`](mdtensor/logic/greater.hpp), [`greater_equal`](mdtensor/logic/greater_equal.hpp), [`less`](mdtensor/logic/less.hpp), [`less_equal`](mdtensor/logic/less_equal.hpp), [`equal`](mdtensor/logic/equal.hpp), [`not_equal`](mdtensor/logic/not_equal.hpp) | [`np.all`](https://numpy.org/doc/stable/reference/generated/numpy.all.html), [`np.any`](https://numpy.org/doc/stable/reference/generated/numpy.any.html), [`np.isinf`](https://numpy.org/doc/stable/reference/generated/numpy.isinf.html), [`np.isnan`](https://numpy.org/doc/stable/reference/generated/numpy.isnan.html), [`np.logical_and`](https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html), [`np.logical_or`](https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html), [`np.logical_not`](https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html), [`np.logical_xor`](https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html), [`np.allclose`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html), [`np.isclose`](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html), [`np.array_equal`](https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html), [`np.array_equiv`](https://numpy.org/doc/stable/reference/generated/numpy.array_equiv.html), [`np.greater`](https://numpy.org/doc/stable/reference/generated/numpy.greater.html), [`np.greater_equal`](https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html), [`np.less`](https://numpy.org/doc/stable/reference/generated/numpy.less.html), [`np.less_equal`](https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html), [`np.equal`](https://numpy.org/doc/stable/reference/generated/numpy.equal.html), [`np.not_equal`](https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html) |
| Math | [`sin`](mdtensor/math/sin.hpp), [`cos`](mdtensor/math/cos.hpp), [`tan`](mdtensor/math/tan.hpp), [`atan2`](mdtensor/math/atan2.hpp), [`deg2rad`](mdtensor/math/deg2rad.hpp), [`rad2deg`](mdtensor/math/rad2deg.hpp), [`sum`](mdtensor/math/sum.hpp), [`add`](mdtensor/math/add.hpp), [`negative`](mdtensor/math/negative.hpp), [`multiply`](mdtensor/math/multiply.hpp), [`divide`](mdtensor/math/divide.hpp), [`subtract`](mdtensor/math/subtract.hpp), [`maximum`](mdtensor/math/maximum.hpp), [`max`](mdtensor/math/max.hpp), [`nanmax`](mdtensor/math/nanmax.hpp), [`minimum`](mdtensor/math/minimum.hpp), [`min`](mdtensor/math/min.hpp), [`nanmin`](mdtensor/math/nanmin.hpp), [`clip`](mdtensor/math/clip.hpp), [`sqrt`](mdtensor/math/sqrt.hpp), [`absolute`](mdtensor/math/absolute.hpp), [`sign`](mdtensor/math/sign.hpp) | [`np.sin`](https://numpy.org/doc/stable/reference/generated/numpy.sin.html), [`np.cos`](https://numpy.org/doc/stable/reference/generated/numpy.cos.html), [`np.tan`](https://numpy.org/doc/stable/reference/generated/numpy.tan.html), [`np.atan2`](https://numpy.org/doc/stable/reference/generated/numpy.atan2.html), [`np.deg2rad`](https://numpy.org/doc/stable/reference/generated/numpy.deg2rad.html), [`np.rad2deg`](https://numpy.org/doc/stable/reference/generated/numpy.rad2deg.html), [`np.sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html), [`np.add`](https://numpy.org/doc/stable/reference/generated/numpy.add.html), [`np.negative`](https://numpy.org/doc/stable/reference/generated/numpy.negative.html), [`np.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html), [`np.divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html), [`np.subtract`](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html), [`np.maximum`](https://numpy.org/doc/stable/reference/generated/numpy.maximum.html), [`np.max`](https://numpy.org/doc/stable/reference/generated/numpy.max.html), [`np.nanmax`](https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html), [`np.minimum`](https://numpy.org/doc/stable/reference/generated/numpy.minimum.html), [`np.min`](https://numpy.org/doc/stable/reference/generated/numpy.min.html), [`np.nanmin`](https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html), [`np.clip`](https://numpy.org/doc/stable/reference/generated/numpy.clip.html), [`np.sqrt`](https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html), [`np.absolute`](https://numpy.org/doc/stable/reference/generated/numpy.absolute.html), [`np.sign`](https://numpy.org/doc/stable/reference/generated/numpy.sign.html) |
| Linear algebra | [`matmul`](mdtensor/linalg/matmul.hpp), [`matvec`](mdtensor/linalg/matvec.hpp), [`vecmat`](mdtensor/linalg/vecmat.hpp), [`cholesky`](mdtensor/linalg/cholesky.hpp), [`norm`](mdtensor/linalg/norm.hpp), [`inv`](mdtensor/linalg/inv.hpp), [`lu`](mdtensor/linalg/lu.hpp), [`solve`](mdtensor/linalg/solve.hpp) | [`np.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html), [`np.matvec`](https://numpy.org/doc/stable/reference/generated/numpy.matvec.html), [`np.vecmat`](https://numpy.org/doc/stable/reference/generated/numpy.vecmat.html), [`np.linalg.cholesky`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html), [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html), [`np.linalg.inv`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html), [`scipy.linalg.lu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu.html), [`np.linalg.solve`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html) |
| Random | [`rand`](mdtensor/random/rand.hpp), [`randint`](mdtensor/random/randint.hpp), [`uniform`](mdtensor/random/uniform.hpp) | [`np.random.rand`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html), [`np.random.randint`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html), [`np.random.uniform`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html) |
| Utility | [`broadcast_extents`](mdtensor/util/broadcast_extents.hpp), [`fill`](mdtensor/util/fill.hpp) | [`np.broadcast_shapes`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_shapes.html), [`ndarray.fill`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.fill.html) |

---

## 설치

mdtensor는 header-only library로, repository를 include path에 추가한 뒤 다음 header를 include하면 바로 사용 가능합니다.
단, C++20 이상의 컴파일러가 요구됩니다.

```cpp
#include "mdtensor/mdtensor.hpp"
```

---

## 테스트 및 benchmark

저장소는 [Bazel](https://bazel.build/) 빌드 시스템을 사용합니다. Bazel을 설치한 뒤 아래 명령어로 테스트와 벤치마크를 실행할 수 있습니다. 또한 저장소의 `.devcontainer/dockerfile`로 구성한 개발 환경에서는 별도 설정 없이 바로 실행할 수 있습니다.

### 1. 전체 테스트 실행:

```bash
bazel test tests/...
```

### 2. 벤치마크 실행 (예):

```bash
bazel run benchmarks/math/add:main
```

---

## Roadmap

- NumPy API coverage 확장.
- General broadcasting rule 강화.
- Lazy computation 도입.
- Compile-time 및 run-time 계산량 분석을 통한 최적 backend 선택 (예: SIMD, Eigen, CPU/GPU multiprocessor 등).

---

## Contributing

이 프로젝트는 현재 적극적으로 관리되고 있으나, 신규 API 추가와 framework 수정은 저자의 필요를 중심으로 진행되고 있습니다. 새로운 API, framework 개선, backend 확장 등을 함께 개발하고 싶은 분들은 issue 또는 pull request를 편하게 생성해 주세요.

---

## License

mdtensor는 Apache License 2.0으로 배포됩니다. 자세한 내용은 [`LICENSE`](LICENSE)를 참고하세요.

mdtensor는 CTMD v0.16.1 (2025.08.19 release)에서 유래한 일부 코드를 포함합니다. CTMD는 Uon Robotics에서 본 저자에 의해 개발된 Apache License 2.0 기반 라이브러리입니다. 이후 mdtensor는 프레임워크 변경, API 확장 구현 등 크게 수정되었습니다.

CTMD의 원저작권은 Uon Robotics에 있으며, mdtensor의 수정 및 확장 부분은 Chan-Soon Lim에 의해 유지 관리됩니다.
