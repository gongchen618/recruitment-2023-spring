#pragma once

#include "Matrices.hh"

Matrix transpose(const Matrix& A);
Matrix empty_matrix(int m, int n);
void print(const Matrix& A);

SparseMatrix transpose_sparse(const SparseMatrix& A);
void print_sparse(const SparseMatrix& A);
