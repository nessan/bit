# README

This directory contains the source for some simple command line programs that exercise various features of the `bit` library.
Each file is stand-alone.
For example, the file `matrix01.cpp` will compile into the executable `matrix01`.

## The Build Process

The `CMakeLists.txt` build file at the top level of the repo adds all these little example programs as targets if you happen to be extending/building the library (as opposed to just using it where these examples are of no direct interest).
It automatically adds & resolves the dependency on the `bit` header-only library.

## Other Dependencies

Many of these programs also depend on the header-only `utilities::utilities` library of useful functions and classes (such as a handy `stopwatch` class, etc.). The `CMakeLists.txt` build file at the top level of the repo will also automatically resolve and add that dependency for you.

## The `charpoly.py` File

The `bit::matrix` class has an implementation of Danilevsky's method to compute characteristic polynomials for matrices over GF(2).
We want to test the results from that code against some “gold standard,” which we produce from the Python file `charpoly.py`.
The well-known, well-tested Python package `Sympy` has a Matrix class with a `charpoly()` method to compute characteristic polynomials.
While `Sympy` doesn't have any direct support for matrices, specifically over GF(2), it does support integer-based matrices.
So, we build integer matrices with elements that are all either 0 or 1 and then extract their characteristic polynomials.
Those will have arbitrary integer coefficients, but if you take each modulo 2, you get the equivalent characteristic polynomial over GF(2).

The Python program `charpoly.py` creates binary matrices in `Sympy` format over a range of sizes and then calls `charpoly()` to get characteristic polynomials.
It then converts those to the `bit` library equivalent.
Each matrix & its characteristic polynomial is written to a data file (along with some comments) in a form that we can read to create a `bit::matrix` and a `bit::vector`.
We can then check that our `C++` `characteristic_polynomial()` method returns the same results as that `bit::vector`.

NOTE: Computing characteristic polynomials for even modest integer matrices will quickly run into huge numbers (beyond the capacity of 64-bit integers) -- though this isn't an issue for `Sympy`, and it will be dog slow to get any results!
This is true even if you start with matrices where the elements are all zeros or ones.
