#include "SpMM_utils.hh"

#include <iostream>

Matrix transpose(const Matrix& A) {
  int A_rows, A_cols;
  std::tie(A_rows, A_cols) = A.size();
  auto T = Matrix(A_cols, A_rows);
  for (int i = 0; i < A_rows; ++i) {
    for (int j = 0; j < A_cols; ++j) {
      T.at(j, i) = A.at(i, j);
    }
  }
  return T;
}

Matrix empty_matrix(int m, int n) {
  vector<float> T(m * n);
  return Matrix(T, m, n);
}

void print(const Matrix& A) {
  int A_rows, A_cols;
  std::tie(A_rows, A_cols) = A.size();
  for (int i = 0; i < A_rows; ++i) {
    for (int j = 0; j < A_cols; ++j) {
      printf("%5.4f ", A.at(i, j));
    }
    puts("");
  }
}

SparseMatrix transpose_sparse(const SparseMatrix& A) {
  int A_rows, A_cols;
  std::tie(A_rows, A_cols) = A.size();
  auto T = SparseMatrix(A_cols, A_rows);
  T.fill_random();
  for (int i = 0; i < A_rows; ++i) {
    for (int j = 0; j < A_cols; ++j) {
      T.at(j, i) = A.at(i, j);
    }
  }
  return T;
}

void print_sparse(const SparseMatrix& A) {
  int A_rows, A_cols;
  std::tie(A_rows, A_cols) = A.size();
  for (int i = 0; i < A_rows; ++i) {
    for (int j = 0; j < A_cols; ++j) {
      printf("%5.4f ", A.at(i, j));
    }
    puts("");
  }
}
