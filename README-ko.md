[한국어](./README-ko.md)
| [English](./README.md)

# mdtensor — NumPy-like 문법을 지향하는 Modern C++ Tensor Library

**mdtensor**는 [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan)(C++23) 기반 view와 [`kokkos::mdarray`](https://github.com/kokkos/mdspan/blob/stable/include/experimental/mdarray) 기반 container를 중심으로 설계된 header-only Modern C++ 텐서 라이브러리입니다. 목표는 NumPy에 가까운 텐서 프로그래밍 경험을 C++에서 제공하면서도, C++ 개발자가 중요하게 여기는 정적 타입 정보, 가능한 범위의 `constexpr` 계산, 낮은 오버헤드의 view, 사용자 소유 메모리와의 직접 연동성을 유지하는 것입니다.

---

## 왜 mdtensor인가?

NumPy는 Python 레벨의 반복문과 함수 호출 오버헤드를 줄이기 위해 vectorized operation과 broadcasting을 적극적으로 활용하도록 발전해 왔습니다. 다차원 연산을 하나의 배열 연산으로 표현하면 반복문을 라이브러리 내부로 밀어 넣을 수 있고, 메모리 접근 패턴도 더 예측 가능해져 현대 CPU에서 높은 성능을 얻기 쉽습니다. 반면 C++에서는 함수 호출 오버헤드가 상대적으로 작고, 실무에서는 Eigen과 같은 행렬 중심 라이브러리가 널리 사용되어 왔기 때문에 NumPy 스타일의 일반적인 N차원 텐서 연산 라이브러리에 대한 수요가 비교적 작았습니다.

C++23에 추가된 [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan)은 C++에서도 다차원 view를 표준적인 방식으로 표현하려는 중요한 시도입니다. 또한 [`std::linalg`](https://en.cppreference.com/w/cpp/numeric/linalg)는 C++26에 추가될 예정입니다. `std::mdspan`은 rank가 compile-time에 결정되어야 한다는 제약이 있지만, zero-overhead abstraction을 목표로 설계되어 C++에서 다차원 연산을 효율적으로 표현하기에 적합합니다.

저자는 로보틱스 알고리즘을 연구하며 Python 기반 알고리즘 코드를 C++ 또는 C로 옮겨 PC와 MCU 환경에서 고속화하는 작업을 오랫동안 수행해 왔습니다. Python에도 PyPy, Numba 등 다양한 고속화 도구가 있지만, 성능과 배포 환경의 제약 때문에 결국 C++에서 알고리즘을 다시 구현하고 Python 결과와 일치시키는 과정이 반복되는 경우가 많았습니다. 여기에 SIMD, CPU/GPU 병렬화까지 적용하려면 매번 별도의 최적화 작업이 필요했습니다.

최근에는 SIMD 기반 계산을 통해 고성능 알고리즘을 microseconds 단위에서 수행하려는 연구도 활발히 진행되고 있습니다. 예를 들어 [Motions in Microseconds via Vectorized Sampling-Based Planning](https://arxiv.org/abs/2309.14545)은 vectorized computation이 로보틱스 알고리즘 성능에 미치는 가능성을 잘 보여줍니다. mdtensor는 바로 이 영역, 즉 **C++에서 NumPy-like 다차원 연산을 정적 타입 정보와 낮은 오버헤드로 표현하는 것**을 목표로 합니다.

mdtensor는 다음을 제공합니다.

- **NumPy-like API**: NumPy에서 익숙한 element-wise math, logic, manipulation, broadcasting 스타일의 API.
- **[`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan) 기반 view**: 외부 메모리를 소유하지 않고 참조할 수 있는 경량 다차원 view.
- **[`kokkos::mdarray`](https://github.com/kokkos/mdspan/blob/stable/include/experimental/mdarray) 기반 container**: std::mdspan 생태계에서 함께 개발된 owning multidimensional container.
- **`constexpr` 지향 구현**: 가능한 많은 연산을 compile-time 테스트 또는 계산에 사용할 수 있도록 설계.
- **Zero-copy broadcasting/manipulation view**: 가능한 경우 eager copy 대신 stride 기반 view 변환 사용.

---

## 개발 철학

### 1. NumPy-like 표면 API, C++-native 내부 구조

mdtensor는 `add`, `multiply`, `allclose`, `broadcast_to`, `reshape`, `transpose`, `matmul`, `inv`, `lu`, `solve`처럼 NumPy에서 익숙한 이름과 의미를 차용합니다. 그러나 내부 구현은 C++의 타입 시스템과 [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan) 중심의 view abstraction에 맞춰 구성됩니다. scalar, mdspan, mdarray, 그리고 core conversion utility로 변환 가능한 객체를 대상으로 동작합니다.

### 2. View 우선, copy는 필요한 경우에만

입력과 변환 과정에서는 mdspan view를 우선 사용합니다. Broadcasting은 stride 0 dimension으로 표현하고, transpose와 reshape는 메모리 layout이 허용하는 경우 view 연산으로 처리하도록 설계합니다. 새로운 결과 객체가 필요한 out-of-place API에서만 owning mdarray 스타일 결과를 반환합니다.

### 3. 최대한의 constexpr, runtime에서는 zero-overhead abstraction을 핵심 목표로 둠

많은 테스트와 예제는 tensor operation을 constant-evaluation context에서 검증할 수 있도록 작성됩니다. 이를 위해 구현은 작은 loop, constexpr helper, static extent propagation, core path에서 runtime-only construct를 피하는 방향으로 구성됩니다. 또한 성능상의 이점이 있다면 최신 C++의 기능도 적극적으로 활용합니다.

### 4. Static shape 정보 활용

mdtensor는 C++23 [`std::extents`](https://en.cppreference.com/w/cpp/container/mdspan/extents)를 활용합니다. compile-time에 결정 가능한 크기는 extents의 static extent로 보존하고, runtime에만 알 수 있는 크기는 dynamic extent로 다룹니다. 이를 통해 static shape가 있는 경우에는 더 많은 검증과 최적화를 compile-time에 수행할 수 있습니다.

### 5. 작은 kernel을 조합해 큰 알고리즘을 구성

core의 `batch`/`batch_out` 메커니즘은 scalar 또는 low-rank element kernel을 N차원 텐서 연산으로 확장하기 위한 기반입니다. Element-wise math, logical comparison, reduction, linear algebra wrapper는 이 재사용 가능한 building block 위에 구성됩니다.

### 6. Backend는 선택 사항이고 API의 중심은 아님

mdtensor는 Eigen, OpenMP 같은 optional backend를 사용할 수 있지만, API 자체가 특정 backend에 종속되지는 않습니다. 기본 라이브러리는 header-only이고 가볍게 유지됩니다.

---

## 빠른 예제 ([오른쪽 링크를 누르면 Godbolt에서 실행 가능합니다 😃](https://godbolt.org/z/cYea11TK4))

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

Output:

```bash
a extents: (3, 1, 2)
a: [[[1, 1]], [[1, 1]], [[1, 1]]]

b extents: (2, 1)
b: [[2], [2]]

c extents: (3, 2, 2)
c: [[[3, 3], [3, 3]], [[3, 3], [3, 3]], [[3, 3], [3, 3]]]
```

이 예제는 mdtensor의 핵심 아이디어를 보여줍니다. NumPy-like broadcasting을 C++ static extents와 compile-time validation 안에서 사용할 수 있습니다.

---

## API 그룹

현재까지 구현된 public API는 다음 module들로 구성됩니다. 표의 “유사 NumPy/SciPy 함수”는 API 이름과 의미상 대응 관계를 나타내며, 완전한 동작 동일성을 보장한다는 의미는 아닙니다.

| Module | API | 유사 NumPy/SciPy 함수 |
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

mdtensor는 header-only library입니다. repository를 include path에 추가한 뒤 다음 header를 include하면 사용할 수 있습니다.

```cpp
#include "mdtensor/mdtensor.hpp"
```

---

## 테스트 및 benchmark

저장소는 [Bazel](https://bazel.build/) 빌드 시스템을 사용합니다. Bazel을 설치한 뒤 아래 command로 테스트와 벤치마크를 실행할 수 있습니다. 또한 저장소의 `.devcontainer/dockerfile`로 구성한 개발 환경에서는 별도 설정 없이 바로 실행할 수 있습니다.

테스트 실행:

```bash
bazel test tests/...
```

벤치마크 실행 (예제):

```bash
bazel run benchmarks/math/add:main
```

---

## 개발 Roadmap

향후 개발 방향은 다음과 같습니다.

- NumPy API coverage 확장.
- General broadcasting rule 강화.
- Lazy computation 도입.
- Compile-time 및 run-time 계산량 분석을 통한 최적 backend 선택 (예: SIMD, Eigen, CPU/GPU multiprocessor 등).

---

## Contributing

이 프로젝트는 적극적으로 관리되고 있습니다. 다만 신규 API 추가와 framework 수정은 현재 저자의 필요를 중심으로 진행되고 있습니다. 새로운 API, framework 개선, backend 확장 등을 함께 개발하고 싶은 분들은 issue 또는 pull request를 편하게 생성해 주세요.

---

## License

mdtensor는 Apache License 2.0으로 배포됩니다. 자세한 내용은 [`LICENSE`](LICENSE)를 참고하세요.

mdtensor는 CTMD v0.16.1 (2025.08.19 release)에서 유래한 일부 코드를 포함합니다. CTMD는 Uon Robotics에서 본 저자에 의해 개발된 Apache License 2.0 기반 라이브러리입니다. 이후 mdtensor는 더 넓은 NumPy-like tensor API, `std::mdspan`/`mdarray` 기반 view semantics, 확장된 broadcasting 및 manipulation 기능, logic/math/random utility, 그리고 LU decomposition, solve, Cholesky decomposition 등 선형대수 기능을 포함하도록 크게 수정·재구성·확장되었습니다.

CTMD의 원저작권은 Uon Robotics에 있으며, mdtensor의 수정 및 확장 부분은 Chan-Soon Lim이 유지 관리합니다.
