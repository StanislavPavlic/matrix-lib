#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-too-small-loop-variable"
#include <iostream>
#include "matrix/matrix.hpp"

using namespace matrix;

using mat_t = Matrix<double>;

int main() {
    std::cout << "Example 1" << std::endl;
    mat_t A = {
            {1, 0},
            {0, 1}
    };
    std::cout << "A =" << A << std::endl;
    std::cout << "A == 3 * A / 3 ? " << ((A == (3.0 * A / 3.0)) ? "true" : "false") << std::endl;



    std::cout << std::endl << std::endl << std::endl << "Example 2" << std::endl;
    A = {
            {3,  9,  6},
            {4, 12, 12},
            {1, -1,  1}
    };
    mat_t b = {
            {12},
            {12},
            { 1}
    };
    std::cout << "A =" << A << std::endl << "b =" << b << std::endl;
    try {
        auto solution = solve_lu(A, b);
        std::cout << "LU solution:" << solution << std::endl;
    } catch (...) {
        std::cout << "Can't solve with LU" << std::endl;
    }
    try {
        auto solution = solve_lup(A, b);
        std::cout << "LUP solution:" << solution << std::endl;
    } catch (...) {
        std::cout << "Can't solve with LUP" << std::endl;
    }



    std::cout << std::endl << std::endl << "Example 3" << std::endl;
    A = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
    };
    std::cout << "A =" << A << "Determinant: " << A.det() << std::endl << std::endl;
    try {
        auto solution = A.lu();
        std::cout << "LU:" << solution << std::endl;
    } catch (...) {
        std::cout << "Can't decompose with LU" << std::endl;
    }
    try {
        auto solution = A.lup().first;
        std::cout << "LUP:" << solution << std::endl;
    } catch (...) {
        std::cout << "Can't decompose with LUP" << std::endl;
    }
    b = {
            {1},
            {1},
            {1}
    };
    try {
        auto solution = solve_lu(A, b);
        std::cout << "LU solution:" << solution << std::endl;
    } catch (...) {
        std::cout << "Can't solve with LU" << std::endl;
    }
    try {
        auto solution = solve_lup(A, b);
        std::cout << "LUP solution:" << solution << std::endl;
    } catch (...) {
        std::cout << "Can't solve with LUP" << std::endl;
    }



    std::cout << std::endl << std::endl << std::endl << "Example 4" << std::endl;
    A = {
            {      0.000001, 3000000, 2000000},
            {1000000       , 2000000, 3000000},
            {2000000       , 1000000, 2000000}
    };
    std::cout << "A =" << A << std::endl;
    b = {
            {12000000.000001},
            {14000000       },
            {10000000       }
    };
    std::cout << "b =" << b << std::endl;
    try {
        auto solution = solve_lu(A, b);
        std::cout << "LU solution:" << solution << std::endl;
    } catch (...) {
        std::cout << "Can't solve with LU" << std::endl;
    }
    try {
        auto solution = solve_lup(A, b);
        std::cout << "LUP solution:" << solution << std::endl;
    } catch (...) {
        std::cout << "Can't solve with LUP" << std::endl;
    }



    std::cout << std::endl << std::endl << "Example 5" << std::endl;
    A = {
            {0, 1, 2},
            {2, 0, 3},
            {3, 5, 1}
    };
    std::cout << "A =" << A << std::endl;
    b = {
            {6},
            {9},
            {3}
    };
    std::cout << "b =" << b << std::endl;
    try {
        auto solution = solve_lup(A, b);
        std::cout << "LUP solution:" << solution << std::endl;
    } catch (...) {
        std::cout << "Can't solve with LUP" << std::endl;
    }



    std::cout << std::endl << std::endl << "Example 6" << std::endl;
    A = {
            {4000000000           , 1000000000           , 3000000000           },
            {         4           ,          2           ,          7           },
            {         0.0000000003,          0.0000000005,          0.0000000002}
    };
    std::cout << "A =" << A << std::endl;
    b = {
            {9000000000           },
            {        15           },
            {         0.0000000015}
    };
    std::cout << "b =" << b << std::endl;
    try {
        auto solution = solve_lup(A, b);
        std::cout << "LUP solution:" << solution << std::endl;
    } catch (...) {
        std::cout << "Can't solve with LUP" << std::endl;
    }
    try {
        for (int i = 0; i < A.rows(); ++i) {
            double avg = 0;
            int count = 0;
            for (int j = 0; j < A.cols(); ++j) {
                ++count;
                avg += (A[i][j] - avg) / count;
            }
            for (int j = 0; j < A.cols(); ++j) {
                A[i][j] /= avg;
            }
            b[i][0] /= avg;
        }
        auto solution = solve_lup(A, b);
        std::cout << "LUP solution (scaled):" << solution << std::endl;
    } catch (...) {
        std::cout << "Can't solve (scaled) with LUP" << std::endl;
    }



    std::cout << std::endl << std::endl << "Example 7" << std::endl;
    A = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
    };
    std::cout << "A =" << A << std::endl;
    try {
        auto solution = ~A;
        std::cout << "Inv(A) =" << solution << std::endl;
    } catch (...) {
        std::cout << "Can't calculate inverse of A with LUP" << std::endl;
    }



    std::cout << std::endl << std::endl << std::endl << "Example 8" << std::endl;
    A = {
            { 4, -5, -2},
            { 5, -6, -2},
            {-8,  9,  3}
    };
    std::cout << "A =" << A << std::endl;
    try {
        auto solution = ~A;
        std::cout << "Inv(A) =" << solution << std::endl;
    } catch (...) {
        std::cout << "Can't calculate inverse of A with LUP" << std::endl;
    }



    std::cout << std::endl << std::endl << "Example 9" << std::endl;
    A = {
            { 4, -5, -2},
            { 5, -6, -2},
            {-8,  9,  3}
    };
    std::cout << "A =" << A << std::endl;
    try {
        auto solution = A.det();
        std::cout << "det(A) = " << solution << std::endl;
    } catch (...) {
        std::cout << "Can't calculate determinant of A with LUP" << std::endl;
    }



    std::cout << std::endl << std::endl << std::endl << "Example 10" << std::endl;
    A = {
            {3,  9,  6},
            {4, 12, 12},
            {1, -1,  1}
    };
    std::cout << "A =" << A << std::endl;
    try {
        auto solution = A.det();
        std::cout << "det(A) = " << solution << std::endl;
    } catch (...) {
        std::cout << "Can't calculate determinant of A with LUP" << std::endl;
    }

    return 0;
}

#pragma clang diagnostic push
