#include <bit/bit.h>
int main()
{
    std::size_t m = 12;

    auto A = bit::matrix<>::random(m);
    auto b = bit::vector<>::random(m);
    auto x = bit::solve(A, b);

    if (x) {
        // Check that x is indeed a solution by computing A.x and comparing that to b
        auto Ax = bit::dot(A, *x);
        std::cout << "bit::matrix A, solution vector x, product A.x, and right hand side b:\n";
        std::cout << "      A         x      A.x      b\n";
        bit::print(A, *x, Ax, b);
        std::cout << "So A.x == b? " << (Ax == b ? "YES" : "NO") << '\n';
    }
    else {
        std::cout << "System A.x = b has NO solutions for A and b as follows:\n";
        std::cout << "      A         x\n";
        bit::print(A, b);
    }
}