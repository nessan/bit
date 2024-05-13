# README

This repository holds the source code and documentation for `bit`, a header-only C++ library for performing linear algebra in bit space. The library represents bit vectors by `bit::vector` and bit matrices by `bit::matrix`. In professional mathematics jargon, the classes allow you to perform linear algebra over GF(2), the simplest Galois field with just two elements, 0 and 1.

GF(2) keeps things closed in $\{0, 1\}$ by performing arithmetic operations mod 2. Addition/subtraction becomes the `XOR` operation, while multiplication/division becomes the `AND` operation. The `bit` library uses those equivalences to efficiently perform most interactions on and between bit-vectors and bit-matrices by simultaneously working on whole blocks of elements.

The `bit` library provides a rich interface to set up and manipulate bit-vectors and bit-matrices in various ways. Amongst other things, the interface includes methods to solve systems of linear equations over GF(2) and to look at the eigen-structure of bit-matrices.

The bit-vector and bit-matrix classes are efficient and pack the individual bit elements into natural word blocks. You can size/resize these dynamic objects as needed at run-time.

## Installation

This library is header-only, so there is nothing to compile and link—drop the `bit` include directory somewhere convenient, and you're good to go.

Alternatively, if you are using `CMake`, you can use the standard `FetchContent` module by adding a few lines to your project's `CMakeLists.txt` file:

```cmake
include(FetchContent)
FetchContent_Declare(bit URL https://github.com/nessan/bit/releases/download/current/bit.zip)
FetchContent_MakeAvailable(bit)
```

This command downloads and unpacks an archive of the current version of `bit` to your project's build folder. You can then add a dependency on `bit::bit`, a `CMake` alias for `bit`. `FetchContent` will automatically ensure the build system knows where to find the downloaded header files and any needed compiler flags.

Used like this, `FetchContent` will only download a minimal library version without any redundant test code, sample programs, documentation files, etc.

## Example

Here is a simple example of a program that uses `bit`:

```cpp
#include <bit/bit.h>
int main()
{
    auto M = bit::matrix<>::random(6,6);
    auto c = bit::characteristic_polynomial(M);
    std::cout << "The bit-matrix M:\n" << M << "\n";
    std::cout << "has characteristic polynomial c(x) = "
              << c.to_polynomial() << ".\n";
    std::cout << "The matrix polynomial sum c(M):\n";
    std::cout << bit::polynomial_sum(c, M) << "\n";
    return 0;
}
```

This program creates a random 6 x 6 bit-matrix `M` where 0 & 1 are equally likely to occur and then extracts its characteristic polynomial $c(x) = c_0 + c_1 x + c_2 x^2 + ... + c_6 x^6$. Finally, the program verifies that `M` satisfies its characteristic equation as expected from the Cayley-Hamilton theorem.

Here is the output from one run of the program:

```sh
The bit-matrix M:
│0 1 1 0 1 0│
│0 1 1 0 1 1│
│1 0 0 1 0 1│
│0 1 1 0 0 1│
│1 0 0 1 1 1│
│0 0 0 1 1 0│
has characteristic polynomial c(x) = x^2 + x^3 + x^6.
The polynomial sum of the bit-matrix c(M) should be all zeros:
│0 0 0 0 0 0│
│0 0 0 0 0 0│
│0 0 0 0 0 0│
│0 0 0 0 0 0│
│0 0 0 0 0 0│
│0 0 0 0 0 0│
```

**NOTE:** `bit` makes it possible to quickly extract the characteristic polynomial for a bit-matrix with millions of elements -- ​a problem that chokes a naive implementation that does not consider the unique nature of arithmetic in GF(2).

## Why Use this Library?

The standard library already has `std::bitset`, an efficient _bitset_ class. Recognizing that `std::bitset` is familiar and well thought through, `bit::vector` replicates and extends much of that interface.

All `std::bitset` objects have a fixed size determined at compile time. The well-known _Boost_ library adds a dynamic version `boost::dynamic_bitset`, where the bitset size can be set and changed at runtime.

However, as the two names suggest, those types are aimed at _bitsets_ instead of _bit-vectors_. So, for example, they print in _bit-order_ with the least significant element/bit on the right. More importantly, those classes don't have any particular methods aimed at linear algebra, and neither does the standard library's vector class `std::vector`.

On the other hand, several well-known linear algebra libraries, such as [Eigen](https://eigen.tuxfamily.org/overview.php?title=Main_Page), exist. Those packages efficiently manage all the standard _numeric_ types (floats, doubles, integers, etc.) but do not correctly handle GF(2). You can create matrices of integers where all the elements are 0 or 1, but there is no built-in knowledge in those libraries that arithmetic is mod 2.

For example, you might use `Eigen` to create an integer matrix of all 0's and 1's and then use a built-in function from that library to extract the characteristic polynomial. Modding the coefficients of that polynomial with 2 gets the appropriate version for GF(2). Technically, this works, but you will have overflow problems for even relatively modest-sized matrices with just a few hundred rows and columns. Of course, you might use an underlying `BitInt` type that never overflows, but the calculations become dog slow for larger bit-matrices, which doesn't help much.

This specialized `bit` library is better for linear algebra problems over GF(2). Consider it if, for example, your interest is in cryptography or random number generation.

## Documentation

You can read the project's documentation [here](https://nessan.github.io/bit/). \
We used the static website generator [Quarto](https://quarto.org) to construct the documentation site.

### Contact

You can contact me by email [here](mailto:nzznfitz+gh@icloud.com).

### Copyright and License

Copyright (c) 2022-present Nessan Fitzmaurice. \
You can use this software under the [MIT license](https://opensource.org/license/mit).
