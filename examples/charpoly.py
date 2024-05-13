#!/usr/bin/env python3

# Our C++ bit::matrix class has an implementation of Danilevsky's method to compute characteristic polynomials
# for matrices over GF(2). That is aware that arithmetic in GF2 is done mod 2 so is efficient and can quite easily
# handle matrices with millions of entries without issue.
#
# We want to test the results from there against some sort of "gold standard" which is what we produce here.
#
# The well known, and well tested Python Sympy package has Matrix class with a charpoly() method to compute
# characteristic polynomials. While it hasn't any direct support for matrices specifically over GF(2), it does
# support integer based matrices. So we simply build integer Matrices with elements that are all either 0 or 1
# (picked at random where you can set the probability of getting 1's) and then extract their characteristic
# polynomials. Those will have arbitrary integer coefficients but if you take each of those modulo 2 you do
# indeed get the equivalent characteristic polynomial over GF(2)!
#
# We run over a range of sizes, creating a binary matrix of that size and then call on charpoly() to get the
# characteristic polynomial which is converted to the GF2 equivalent. The matrix & that characteristic
# polynomial is written to a data file (along with some useful comments) in a form that is we can read to
# create a Matrix and a Vector. We can then check that our C++ characteristic_polynomial() method
# returns the same results as that Vector.
#
# NOTE: Computing characteristic polynomials for even fairly modest integer matrices will quickly run in to
# huge numbers (beyond the capacity of say 64 bit integers -- though this isn't necessarily an issue for Sympy)
# and it will be dog slow to get any results! This is true even if you start with matrices where the elements
# are all zeros or ones.
#
# SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
# SPDX-License-Identifier: MIT

import errno
import sys
from timeit import default_timer as timer

import numpy
import sympy


def random_bit(prob_one=0.5):
    """Returns a random 0 or 1. Probability of getting 1 is the argument."""
    return numpy.random.binomial(n=1, p=prob_one)


def random_bit_matrix(n, prob_ones=0.5):
    """Returns an n x n Sympy matrix over GF2 (second arg = chance of getting 1's)."""
    return sympy.Matrix(n, n, lambda i, j: random_bit(prob_ones))


def gf2_coeffs_charpoly(mat):
    """Return the characteristic polynomial coefficients of a Sympy matrix mod 2 (GF2)"""
    poly_coeffs = mat.charpoly().all_coeffs()
    gf2_coeffs = [p % 2 for p in poly_coeffs]
    gf2_coeffs.reverse()
    return gf2_coeffs


# Data file ...
default_file = 'charpoly.dat'
file_name = input(
    f"Output file to create [default '{default_file}']: ") or default_file

try:
    f = open(file_name, 'w')
except OSError as e:
    print(
        f"Failed to open '{file_name}' for writing (system error = {e.errno})!", file=sys.stderr)
    sys.exit(e.errno)

# Smallest test matrix to generate ...
default_min = 1
n_min = input(
    f"Smallest matrix to generate [default {default_min}]: ") or default_min
try:
    n_min = abs(int(n_min))
except ValueError:
    print(
        f"Failed to parse '{n_min}' as int so will use default of {default_min}!", file=sys.stderr)
    n_min = default_min

# Largest test matrix to generate
default_max = 40
n_max = input(
    f"Largest matrix to generate [default {default_max}]: ") or default_max
try:
    n_max = abs(int(n_max))
except ValueError:
    print(
        f"Failed to parse '{n_max}' as int so will use default of {default_max}!", file=sys.stderr)
    n_max = default_max

# Make sure the range of matrix sizes is in order
if(n_min > n_max):
    n_min, n_max = n_max, n_min
n_matrices = n_max - n_min + 1

# Probability of getting set bits in those matrices
default_prob = 0.5
prob = input(
    f"Probability of getting set bits in the matrices [default {default_prob}]: ") or default_prob
try:
    prob = float(prob)
except ValueError:
    print(
        f"Failed to parse '{prob}' as a float so will use default of {default_prob}!", file=sys.stderr)
    prob = default_prob

# Clamp the probability silently if it lies outside [0,1] ...
if prob < 0.0:
    prob = 0.0
if prob > 1.0:
    prob = 1.0

# And we're off ...
start_time = timer()
with f:
    # Start the data file with an initial comment
    print(
        f'# File contains {n_matrices} binary matrices and their GF2 characteristic polynomials', file=f)
    print(
        f'# In each case, the matrix elements are set to one with probability {prob}', file=f)
    print('# The data was generated in Python using the charpoly() method from the Sympy package', file=f)
    print(file=f)

    # Generate random binary matrices of sizes 1 x 1 through to max_size x max_size.
    # In each case get the characteristic polynomial over GF2.
    # Dump the matrix and the polynomial coefficients to the file in a readable format.
    # Annotate the output with comments that tell the use what they are looking at.
    for n in range(n_min, n_max + 1):
        mat = random_bit_matrix(n, prob)

        comp_start = timer()
        c = gf2_coeffs_charpoly(mat)
        comp_time = timer() - comp_start

        print(f'# {mat.rows} x {mat.cols} Bit-Matrix:', file=f)
        for r in range(mat.rows):
            print(*mat.row(r), sep='', end='', file=f)
            print(';', end='', file=f)
        print(file=f)

        print(
            f'# Sympy characteristic polynomial coefficients computed in {comp_time:.3f} seconds:', file=f)
        print(*c, sep='', file=f)
        print(file=f)
end_time = timer()
seconds = end_time - start_time

print(
    f"Generated characeristic polynomials for {n_matrices} matrices ({n_min} x {n_min} through {n_max} x {n_max}) in a total of {seconds:.3f} seconds.")
