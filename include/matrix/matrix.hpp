#ifndef MATRIX_H
#define MATRIX_H


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <functional>

namespace matrix {

    constexpr double TOL = 1e-6;

    template<typename T>
    using matrix_t = std::vector<std::vector<T>>;

    template<typename T>
    using row_t = std::vector<T>;

    template<typename T>
    using vec_t = std::vector<T>;

    template<typename T>
    using f_t = std::function<T(T)>;

    template<typename T>
    class Matrix {
    public:
        /**
         * Default matrix constructor
         */
        Matrix() : rows_(0), cols_(0), m_() {};

        /**
         * Square matrix constructor
         * @param size number of rows and columns
         */
        explicit Matrix(std::size_t size) : Matrix(size, size) {}

        /**
         * Default matrix with set dimensions
         * @param rows number of rows
         * @param cols number of columns
         */
        Matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), m_(rows, row_t<T>(cols)) {}

        /**
         * All member constructor
         * @param rows number of rows
         * @param cols number of columns
         * @param m matrix
         */
        Matrix(std::size_t rows, std::size_t cols, const matrix_t<T>& m) : rows_(rows), cols_(cols), m_(m) {}

        /**
         * Construct matrix with certain value
         * @param rows number of rows
         * @param cols number of columns
         * @param val value to be used to fill the matrix
         */
        Matrix(std::size_t rows, std::size_t cols, T val) : rows_(rows), cols_(cols), m_(rows, row_t<T>(cols, val)) {}

        /**
         * Raw matrix representation constructor
         * @param m matrix
         */
        explicit Matrix(const matrix_t<T>& m) : rows_(m.size()), cols_(m.size() ? m[0].size() : 0), m_(m) {}

        /**
         * From file constructor
         * @param filename file to be used for construction
         */
        explicit Matrix(const std::string& filename) : Matrix(read_constr(filename)) {}

        /**
         * Custom matrix constructor
         * @param list vector of vectors in the form of std::initializer_list to be used for construction
         */
        Matrix(std::initializer_list<std::vector<T>> list) : rows_(list.size()),
                                                             cols_(list.size() ? list.begin()->size() : 0), 
                                                             m_(list) {}

        /**
         * Copy constructor
         * @param matrix source matrix to be copied
         */
        Matrix(const Matrix& matrix) : rows_(matrix.rows_), cols_(matrix.cols_), m_(matrix.m_) {}

        /**
         * Default destructor
         */
        ~Matrix() = default;

        /**
         * Returns number of rows in matrix
         * @return number of rows
         */
        std::size_t rows() const {
            return rows_;
        }

        /**
         * Returns number of columns in matrix
         * @return number of columns
         */
        std::size_t cols() const {
            return cols_;
        }

        /**
         * Returns total number of elements in matrix
         * @return total number of elements in matrix
         */
        std::size_t size() const {
            return rows_ * cols_;
        }

        /**
         * Return row at index idx as matrix
         * @param idx index of queried row
         * @return row at index idx as matrix
         */
        Matrix row(int idx) const {
            if (idx < 0 || idx > rows_) {
                throw std::invalid_argument("row index out of bounds");
            }
            return {m_[idx]};
        }

        /**
         * Return column at index idx as matrix
         * @param idx index of queried column
         * @return column at index idx as matrix
         */
        Matrix col(int idx) const {
            if (idx < 0 || idx > cols_) {
                throw std::invalid_argument("column index out of bounds");
            }
            Matrix c(rows_, 1);
            for (int i = 0; i < rows_; ++i) {
                c[i][0] = m_[i][idx];
            }
            return c;
        }

        /**
         * Assignment operator
         * @param matrix source matrix to be assigned
         * @return reference to result matrix
         */
        Matrix& operator=(const Matrix& matrix) {
            if (this == &matrix) {
                return *this;
            }
            rows_ = matrix.rows_;
            cols_ = matrix.cols_;
            m_ = matrix.m_;
            return *this;
        }

        /**
         * Resize matrix to certain size
         * @param r new number of rows
         * @param c new number of columns
         */
        void resize(std::size_t r, std::size_t c) {
            rows_ = r;
            cols_ = c;
            for (auto& col : m_) {
                col.resize(cols_, static_cast<T>(0));
            }
            m_.resize(rows_, std::vector<T>(cols_, static_cast<T>(0)));
        }

        /**
         * Resize matrix to certain number of rows
         * @param r new number of rows
         */
        void resize_rows(std::size_t r) {
            rows_ = r;
            m_.resize(rows_, std::vector<T>(cols_, static_cast<T>(0)));
        }

        /**
         * Resize matrix to certain number of columns
         * @param c new number of columns
         */
        void resize_cols(std::size_t c) {
            cols_ = c;
            for (auto& col : m_) {
                col.resize(cols_, static_cast<T>(0));
            }
        }

        /**
         * Write matrix to text file with given filename/path
         * @param filename filename/path to be written to
         */
        void write(const std::string& filename) const {
            std::ofstream ofs(filename);
            if (ofs.is_open()) {
                for (const auto& row : m_) {
                    for (const auto& el : row) {
                        ofs << el << " ";
                    }
                    ofs << std::endl;
                }
                ofs.close();
            } else {
                std::cerr << "Unable to open file: " << filename << std::endl;
            }
        }

        /**
         * Write matrix to standard output
         */
        void write() const {
            for (const auto& row : m_) {
                for (const auto& el : row) {
                    std::cout << el << " ";
                }
                std::cout << std::endl;
            }
        }

        /**
         * Access operator for matrix
         * @param idx index of row to be accessed
         * @return row at index idx
         */
        row_t<T>& operator[](std::size_t idx) {
            return m_[idx];
        }

        /**
         * Const version of access operator for matrix
         * @param idx index of row to be accessed
         * @return row at index idx
         */
        const row_t<T>& operator[](std::size_t idx) const {
            return m_[idx];
        }

        /**
         * Addition operator for element-wise addition of two matrices
         * @param rhs matrix to be added
         * @return result of addition
         */
        const Matrix operator+(const Matrix& rhs) const {
            if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
                throw std::invalid_argument("matrix dimensions not equal");
            }

            Matrix res(*this);
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    res.m_[i][j] += rhs.m_[i][j];
                }
            }
            return res;
        }

        /**
         * Addition operator for adding numeric value to matrix
         * @tparam S numeric type of value to be added
         * @param rhs numeric value to be added to each element
         * @return result of addition
         */
        template<typename S>
        const Matrix operator+(S rhs) const {
            Matrix res(*this);
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    res.m_[i][j] += rhs;
                }
            }
            return res;
        }

        /**
         * Addition operator for adding numeric value to matrix
         * @tparam S numeric type of matrix elements
         * @tparam V numeric type of value to be added to
         * @param lhs numeric value to be added to each element of rhs
         * @param rhs matrix to which lhs will be added
         * @return result of addition
         */
        template<typename S, typename V>
        friend const Matrix<S> operator+(V lhs, const Matrix<S>& rhs);

        /**
         * Subtraction operator for element-wise subtraction of two matrices
         * @param rhs matrix to be subtracted
         * @return result of subtraction
         */
        const Matrix operator-(const Matrix& rhs) const {
            if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
                throw std::invalid_argument("matrix dimensions not equal");
            }

            Matrix res(*this);
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    res.m_[i][j] -= rhs.m_[i][j];
                }
            }
            return res;
        }

        /**
         * Subtraction operator for subtracting numeric value from matrix
         * @tparam S numeric type of value to be subtracted
         * @param rhs numeric value to be subtracted from each element
         * @return result of subtraction
         */
        template<typename S>
        const Matrix operator-(S rhs) const {
            Matrix res(*this);
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    res.m_[i][j] -= rhs;
                }
            }
            return res;
        }

        /**
         * Subtraction operator for subtracting numeric value to matrix
         * @tparam S numeric type of matrix elements
         * @tparam V numeric type of value to be subtracted from
         * @param lhs numeric value to be subtracted from each element of rhs
         * @param rhs matrix from which lhs will be subtracted
         * @return result of subtraction
         */
        template<typename S, typename V>
        friend const Matrix<S> operator-(V lhs, const Matrix<S>& rhs);

        /**
         * Multiplication operator for dot product of two matrices
         * @param rhs matrix to be multiplied
         * @return dot product
         */
        const Matrix operator*(const Matrix& rhs) const {
            if (cols_ != rhs.rows_) {
                throw std::invalid_argument("matrix dimensions not compatible");
            }

            Matrix res(rows_, rhs.cols_, 0);
            for (int i = 0; i < res.rows_; ++i) {
                for (int j = 0; j < res.cols_; ++j) {
                    for (int k = 0; k < cols_; ++k) {
                        res.m_[i][j] += m_[i][k] * rhs.m_[k][j];
                    }
                }
            }
            return res;
        }

        /**
         * Multiplication operator for element-wise multiplication of matrix with numeric value
         * @tparam S numeric type of value to be multiplied
         * @param rhs numeric value to which will multiply each element
         * @return result of multiplication
         */
        template<typename S>
        const Matrix operator*(S rhs) const {
            Matrix res(*this);
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    res.m_[i][j] *= rhs;
                }
            }
            return res;
        }

        /**
         * Multiplication operator for multiplying numeric value with matrix
         * @tparam S numeric type of matrix elements
         * @tparam V numeric type of value to be multiplied with
         * @param lhs numeric value to be multiplied with each element of rhs
         * @param rhs matrix which will be mutliplied with lhs
         * @return result of multiplication
         */
        template<typename S, typename V>
        friend const Matrix<S> operator*(V lhs, const Matrix<S>& rhs);

        /**
         * Unary minus operator
         * @return matrix multiplied by -1
         */
        const Matrix operator-() const {
            return *this * (-1);
        }

        /**
         * Create matrix from file
         * @param filename path to file which contains the matrix to be created
         * @return matrix
         */
        static Matrix read(const std::string& filename) {
            return Matrix(filename);
        }

        /**
         * Matrix transposition
         * @return transposed matrix
         */
        const Matrix transpose() const {
            Matrix res(cols_, rows_);
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    res.m_[j][i] = m_[i][j];
                }
            }
            return res;
        }

        /**
         * Addition assignment operator
         * @param rhs matrix to be added
         * @return reference to result matrix
         */
        Matrix& operator+=(const Matrix& rhs) {
            if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
                throw std::invalid_argument("matrix dimensions not equal");
            }

            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    this->m_[i][j] += rhs.m_[i][j];
                }
            }
            return *this;
        }

        /**
         * Addition assignment operator
         * @tparam S numeric type for value to be added
         * @param rhs numeric value to be added
         * @return reference to result matrix
         */
        template<typename S>
        Matrix& operator+=(S rhs) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    this->m_[i][j] += rhs;
                }
            }
            return *this;
        }

        /**
         * Subtraction assignment operator
         * @param rhs matrix to be subtracted
         * @return reference to result matrix
         */
        Matrix& operator-=(const Matrix& rhs) {
            if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
                throw std::invalid_argument("matrix dimensions not equal");
            }

            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    this->m_[i][j] -= rhs.m_[i][j];
                }
            }
            return *this;
        }

        /**
         * Subtraction assignment operator
         * @tparam S numeric type for value to be subtracted
         * @param rhs numeric value to be subtracted
         * @return reference to result matrix
         */
        template<typename S>
        Matrix& operator-=(S rhs) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    this->m_[i][j] -= rhs;
                }
            }
            return *this;
        }

        /**
         * Multiplication assignment operator
         * @tparam S numeric type for value to be multiplied by
         * @param rhs numeric value to be multiplied by
         * @return reference to result matrix
         */
        template<typename S>
        Matrix& operator*=(S rhs) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    this->m_[i][j] *= rhs;
                }
            }
            return *this;
        }

        /**
         * Equality operator for checking matrix equality
         * @tparam S type of matrices to be compared
         * @param lhs left matrix
         * @param rhs right matrix
         * @return truth value of equality
         */
        template<typename S>
        friend bool operator==(const Matrix<S>& lhs, const Matrix<S>& rhs);

        bool is_square() const {
            return rows_ == cols_;
        }

        /**
         * Stream operator for output stream
         * @tparam S type of matrix
         * @param os output stream
         * @param rhs matrix
         * @return update output stream
         */
        template<typename S>
        friend std::ostream& operator<<(std::ostream& os, const Matrix<S>& rhs);

        /**
         * In-place LU decomposition of matrix
         * @return LU decomposition matrix
         */
        const Matrix lu() const {
            if (!is_square()) {
                throw std::invalid_argument("matrix not square");
            }

            Matrix res(*this);
            for (int i = 0; i < res.rows_ - 1; ++i) {
                for (int j = i + 1; j < res.cols_; ++j) {
                    if (std::fabs(res.m_[i][i]) < TOL) {
                        throw std::invalid_argument("pivot element near or equal 0");
                    }
                    res.m_[j][i] /= res.m_[i][i];
                    for (int k = i + 1; k < res.rows_; ++k) {
                        res.m_[j][k] -= res.m_[j][i] * res.m_[i][k];
                    }
                }
            }
            return res;
        }

        /**
         * In-place LUP decomposition of matrix
         * @return LUP decomposition matrix, permutation matrix, number of row swaps
         */
        std::pair<Matrix, std::pair<Matrix, int>> lup() const {
            if (!is_square()) {
                throw std::invalid_argument("matrix not square");
            }

            Matrix P(rows_, cols_, 0);
            for (int i = 0; i < rows_; ++i) {
                P[i][i] = 1;
            }
            int count = 0;
            Matrix res(*this);
            for (int i = 0; i < res.rows_ - 1; ++i) {
                int pivot = i;
                for (int j = i + 1; j < res.cols_; ++j) {
                    if (std::abs(res.m_[j][i]) > std::abs(res.m_[pivot][i])) {
                        pivot = j;
                    }
                }
                if (pivot != i) {
                    ++count;
                    std::swap(P[i], P[pivot]);
                    std::swap(res.m_[i], res.m_[pivot]);
                }
                for (int j = i + 1; j < res.cols_; ++j) {
                    if (std::fabs(res.m_[i][i]) < TOL) {
                        throw std::invalid_argument("pivot element near or equal 0");
                    }
                    res.m_[j][i] /= res.m_[i][i];
                    for (int k = i + 1; k < rows_; ++k) {
                        res.m_[j][k] -= res.m_[j][i] * res.m_[i][k];
                    }
                }
            }
            return std::make_pair(res, std::make_pair(P, count));
        }

        /**
         * Forward substitution method for solving systems of linear equations using LU decomposed matrix, L*y=b
         * @param b vector at right-hand side of equations
         * @return vector y
         */
        Matrix fwd_sub(Matrix b) const {
            if (!is_square()) {
                throw std::invalid_argument("matrix not square");
            }
            if (b.cols_ != 1) {
                throw std::invalid_argument("argument is not a vector");
            }
            if (rows_ != b.rows_) {
                throw std::invalid_argument("vector dimension not compatible with matrix");
            }

            for (int i = 0; i < rows_; ++i) {
                for (int j = i + 1; j < cols_; ++j) {
                    b[j][0] -= m_[j][i] * b[i][0];
                }
            }
            return b;
        }

        /**
         * Backward substitution method for solving systems of linear equations using LU decomposed matrix, U*x=y
         * @param b vector from forward substitution method (y)
         * @return vector x (system solution)
         */
        Matrix bwd_sub(Matrix b) const {
            if (!is_square()) {
                throw std::invalid_argument("matrix not square");
            }
            if (b.cols_ != 1) {
                throw std::invalid_argument("argument is not a vector");
            }
            if (rows_ != b.size()) {
                throw std::invalid_argument("vector dimension not compatible with matrix");
            }

            for (int i = rows_ - 1; i >= 0; --i) {
                if (std::fabs(m_[i][i]) < TOL) {
                    throw std::invalid_argument("vector element near or equal 0");
                }
                b[i][0] /= m_[i][i];
                for (int j = 0; j < i; ++j) {
                    b[j][0] -= m_[j][i] * b[i][0];
                }
            }
            return b;
        }

        /**
         * Division operator for 'left division', A^(-1)*b, through LUP decomposition
         * @param b vector at right-hand side of equation Ax=b
         * @return vector x, solution to x=A^(-1)*b
         */
        const Matrix operator/(Matrix b) const {
            auto decomp = lup();
            return decomp.first.bwd_sub(decomp.first.fwd_sub(decomp.second.first * b));
        }

        /**
         * Division operator for dividing each element by given value
         * @tparam S numeric type of value to be divided by
         * @param rhs numeric value to be divided by
         * @return result of division
         */
        template<typename S>
        const Matrix operator/(S rhs) const {
            Matrix res(*this);
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    res.m_[i][j] /= rhs;
                }
            }
            return res;
        }

        /**
         * Division assignment operator for dividing each element by given value
         * @tparam S numeric type of value to be divided by
         * @param rhs numeric value to be divided by
         * @return reference to result matrix
         */
        template<typename S>
        Matrix& operator/=(S rhs) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    this->m_[i][j] /= rhs;
                }
            }
            return *this;
        }

        /**
         * Tilde operator for matrix inversion
         * @return matrix inverse
         */
        const Matrix operator~() const {
            if (!is_square()) {
                throw std::invalid_argument("matrix not square");
            }

            Matrix inv(rows_);
            auto decomp = lup();
            auto P = decomp.second.first;
            for (int i = 0; i < rows_; ++i) {
                Matrix b = P.col(i);
                inv[i] = (decomp.first.bwd_sub(decomp.first.fwd_sub(b))).transpose().m_[0];
            }
            return inv.transpose();
        }

        /**
         * Calculate determinant of matrix using LUP decomposition
         * @return matrix determinant
         */
        T det() const {
            if (!is_square()) {
                throw std::invalid_argument("matrix not square");
            }

            auto decomp = lup();
            double res = decomp.second.second % 2 ? -1 : 1;
            for (int i = 0; i < rows_; ++i) {
                res *= decomp.first[i][i];
            }
            return res;
        }

        /**
         * Calculates the Euclidian norm of a given vector.
         * @return Euclidian norm
         */
        T norm() const {
            if (rows_ != 1 && cols_ != 1) {
                throw std::invalid_argument("not a vector");
            }

            T s = 0;
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    s += m_[i][j] * m_[i][j];
                }
            }

            return sqrt(s);
        }

        /**
         * Multiplication operator for element-wise multiplication of two matrices
         * @param rhs matrix to be mutliplied by
         * @return result of multiplication
         */
        const Matrix mult(const Matrix& rhs) const {
            if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
                throw std::invalid_argument("matrix dimensions not equal");
            }

            Matrix res(*this);
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    res.m_[i][j] *= rhs.m_[i][j];
                }
            }
            return res;
        }

    private:
        /**
         * Text matrix reader
         * @param filename path to file
         * @return matrix in raw form
         */
        matrix_t<T> read_constr(const std::string& filename) {
            std::ifstream ifs(filename);
            std::string line;
            matrix_t<T> mat;
            int i = 0;
            while (std::getline(ifs, line)) {
                std::istringstream iss(line);
                mat.emplace_back(row_t<T>());
                do {
                    T val;
                    if (!(iss >> val)) {
                        break;
                    }
                    mat[i].push_back(val);
                } while (true);
                ++i;
            }
            return mat;
        }

        std::size_t rows_;
        std::size_t cols_;
        matrix_t<T> m_;
    };

    /**
     * Addition operator for adding numeric value to matrix
     * @tparam T numeric type of matrix elements
     * @tparam S numeric type of value to be added to
     * @param lhs numeric value to be added to each element of rhs
     * @param rhs matrix to which lhs will be added
     * @return result of addition
     */
    template<typename T, typename S>
    inline const Matrix<T> operator+(S lhs, const Matrix<T>& rhs) {
        return rhs + lhs;
    }

    /**
     * Subtraction operator for subtracting numeric value to matrix
     * @tparam T numeric type of matrix elements
     * @tparam S numeric type of value to be subtracted from
     * @param lhs numeric value to be subtracted from each element of rhs
     * @param rhs matrix from which lhs will be subtracted
     * @return result of subtraction
     */
    template<typename T, typename S>
    inline const Matrix<T> operator-(S lhs, const Matrix<T>& rhs) {
        return rhs - lhs;
    }

    /**
     * Multiplication operator for multiplying numeric value with matrix
     * @tparam T numeric type of matrix elements
     * @tparam S numeric type of value to be multiplied with
     * @param lhs numeric value to be multiplied with each element of rhs
     * @param rhs matrix which will be mutliplied with lhs
     * @return result of multiplication
     */
    template<typename T, typename S>
    inline const Matrix<T> operator*(S lhs, const Matrix<T>& rhs) {
        return rhs * lhs;
    }

    /**
     * Equality operator for checking matrix equality
     * @tparam T type of matrices to be compared
     * @param lhs left matrix
     * @param rhs right matrix
     * @return truth value of equality
     */
    template<typename T>
    inline bool operator==(const Matrix<T>& lhs, const Matrix<T>& rhs) {
        if (&lhs == &rhs) {
            return true;
        }
        if (lhs.rows_ != rhs.rows_ || lhs.cols_ != rhs.cols_) {
            return false;
        }
        return equals(lhs, rhs, std::is_integral<T>());
    }

    /**
     * Equality method for non-integral types
     * @tparam T non-integral numeric type
     * @param lhs left matrix
     * @param rhs right matrix
     * @return truth value of equality
     */
    template<typename T>
    inline bool equals(const Matrix<T>& lhs, const Matrix<T>& rhs, std::false_type) {
        for (int i = 0; i < lhs.rows(); ++i) {
            for (int j = 0; j < lhs.cols(); ++j) {
                if (std::fabs(lhs[i][j] - rhs[i][j]) > TOL) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Equality method for integral types
     * @tparam T integral numeric type
     * @param lhs left matrix
     * @param rhs right matrix
     * @return truth value of equality
     */
    template<typename T>
    inline bool equals(const Matrix<T>& lhs, const Matrix<T>& rhs, std::true_type) {
        for (int i = 0; i < lhs.rows(); ++i) {
            for (int j = 0; j < lhs.cols(); ++j) {
                if (lhs[i][j] != rhs[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Stream operator for output stream
     * @tparam T type of matrix
     * @param os output stream
     * @param rhs matrix
     * @return update output stream
     */
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Matrix<T>& rhs) {
        os << std::endl;
        for (const auto& row : rhs.m_) {
            for (const auto& el : row) {
                os << el << " ";
            }
            os << std::endl;
        }
        return os;
    }

    /**
     * Solve system of linear equations using LU decomposition, Ax=b => LUx=b
     * @tparam T numeric type
     * @param A matrix representing variable coefficients
     * @param b vector of right-hand side of equations
     * @return solution vector x
     */
    template<typename T>
    Matrix<T> solve_lu(const Matrix<T>& A, Matrix<T> b) {
        auto decomp = A.lu();
        return decomp.bwd_sub(decomp.fwd_sub(b));
    }

    /**
     * Solve system of linear equations using LUP decomposition, Ax=b => PLUx=Pb
     * @tparam T numeric type
     * @param A matrix representing variable coefficients
     * @param b vector of right-hand side of equations
     * @return solution vector x
     */
    template<typename T>
    Matrix<T> solve_lup(const Matrix<T>& A, Matrix<T> b) {
        return A / b;
    }

    /**
     * Calculate absolute values of elements of given matrix
     * @tparam T numeric type
     * @param m input matrix
     * @return matrix with absolute value function applied to its elements
     */
    template<typename T>
    Matrix<T> abs(Matrix<T> m) {
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                m[i][j] = fabs(m[i][j]);
            }
        }
        return m;
    }

    /**
     * Calculate sum of all elements of given matrix
     * @tparam T numeric type
     * @param m input matrix
     * @return sum of all elements
     */
    template<typename T>
    T sum(const Matrix<T>& m) {
        T s = 0;
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                s += m[i][j];
            }
        }
        return s;
    }

    /**
     * Apply function to matrix element-wise
     * @tparam T numeric type
     * @param m input matrix
     * @param f function to be applied
     * @return matrix with the given function f applied to all the elements of the input matrix
     */
    template<typename T>
    Matrix<T> apply(Matrix<T> m, f_t<T> f) {
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                m[i][j] = f(m[i][j]);
            }
        }
        return m;
    }

}

#endif //MATRIX_H
