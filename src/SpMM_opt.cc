#include <immintrin.h>

#include "SpMM.hh"
#include "SpMM_utils.hh"

// cmake --build build; cd build; make test
// default

Matrix SpMM_opt_0(const Matrix &A, const SparseMatrix &B) {
  if (A.size() != B.size()) return Matrix();
  int A_rows, A_cols, B_rows, B_cols;
  vector<float> data;
  std::tie(A_rows, A_cols) = A.size();
  std::tie(B_rows, B_cols) = B.size();
  for (int i = 0; i < A_rows; ++i) {
    for (int j = 0; j < B_rows; ++j) {
      float sum = 0;
      for (int k = 0; k < B_cols; ++k) {
        sum += A.at(i, k) * B.at(j, k);
      }
      data.push_back(sum);
    }
  }
  return Matrix(data, A_rows, B_rows);
}

// COO + simple parallel pragma
// imporvement is little

Matrix SpMM_opt_1(const Matrix &A, const SparseMatrix &B) {
  if (A.size() != B.size()) return Matrix();

  auto AT = transpose(A);
  int AT_rows, AT_cols, B_rows, B_cols;
  std::tie(AT_rows, AT_cols) = AT.size();
  std::tie(B_rows, B_cols) = B.size();

  auto data = empty_matrix(AT_rows, B_rows);

#pragma omp parallel for
  for (int j = 0; j < B_rows; ++j) {
    for (int k = 0; k < B_cols; ++k) {
      if (B.at(j, k) != 0) {
        float v = B.at(j, k);
        for (int i = 0; i < AT_rows; ++i) {
          data.at(j, i) += AT.at(k, i) * v;
        }
      }
    }
  }

  return transpose(data);
}

// CSR + SIMD
// failed for some test points
// but the improvement is little
// so not going to fix

Matrix SpMM_opt_2(const Matrix &A, const SparseMatrix &B) {
  if (A.size() != B.size()) return Matrix();

  auto BT = transpose_sparse(B);
  int A_rows, A_cols, BT_rows, BT_cols;
  std::tie(BT_rows, BT_cols) = BT.size();
  std::tie(A_rows, A_cols) = A.size();

  vector<float> BT_csr_val;
  vector<int> BT_csr_id, BT_csr_col;
  int csr_cnt = 0;
  BT_csr_id.push_back(0);
  for (int i = 0; i < BT_rows; ++i) {
    for (int j = 0; j < BT_cols; ++j) {
      if (BT.at(i, j) != 0) {
        BT_csr_val.push_back(BT.at(i, j));
        BT_csr_col.push_back(j);
        ++csr_cnt;
      }
    }
    BT_csr_id.push_back(csr_cnt);
  }

  vector<float> data(A_rows * BT_cols);
  for (int i = 0; i < A_rows; ++i) {
    for (int j = 0; j < BT_rows; ++j) {
      int l = BT_csr_id[j], r = BT_csr_id[j + 1];
      int k = l;

      /* ---------------- wrote by GPT ---------------------*/
      __m512 a = _mm512_set1_ps(A.at(i, j));
      for (; k + 15 < r; k += 16) {
        __m512i indices = _mm512_loadu_si512(&BT_csr_col[k]);
        __m512 values = _mm512_loadu_ps(&BT_csr_val[k]);
        __m512 result = _mm512_fmadd_ps(
            a, values, _mm512_i32gather_ps(indices, &data[0], sizeof(float)));
        _mm512_i32scatter_ps(&data[0], indices, result, sizeof(float));
      }
      for (; k < r; ++k) {
        float v = BT_csr_val[k];
        int p = BT_csr_col[k];
        data[i * BT_rows + p] += A.at(i, j) * v;
      }
      /* ---------------- wrote by GPT ---------------------*/
    }
  }
  return Matrix(data, A_rows, BT_cols);
}

// blocking

Matrix SpMM_opt(const Matrix &A, const SparseMatrix &B) {
  if (A.size() != B.size()) return Matrix();

  auto BT = transpose_sparse(B);
  int A_rows, A_cols, BT_rows, BT_cols;
  std::tie(BT_rows, BT_cols) = BT.size();
  std::tie(A_rows, A_cols) = A.size();

  auto data = empty_matrix(A_rows, BT_cols);

  int width1 = 64, width2 = 64;
  for (int i = 0; i < BT_rows; i += width1) {
    for (int j = 0; j < A_rows; j += width2) {
      for (int k = 0; k < BT_cols; ++k) {
        for (int m = 0; m < width2; ++m) {
          float sum = 0;
          for (int n = 0; n < width1; ++n) {
            sum += A.at(j + m, i + n) * BT.at(i + n, k);
          }
          data.at(j + m, k) += sum;
        }
      }
    }
  }

  return data;
}


// blocking + OpenMP
/* ---------------- wrote by GPT ---------------------*/

Matrix SpMM_opt(const Matrix &A, const SparseMatrix &B) {
  if (A.size() != B.size()) return Matrix();

  auto BT = transpose_sparse(B);
  int A_rows, A_cols, BT_rows, BT_cols;
  std::tie(BT_rows, BT_cols) = BT.size();
  std::tie(A_rows, A_cols) = A.size();

  auto data = empty_matrix(A_rows, BT_cols);

  int width1 = 64, width2 = 64;
#pragma omp parallel for collapse(2)
  for (int i = 0; i < BT_rows; i += width1) {
    for (int j = 0; j < A_rows; j += width2) {
#pragma omp parallel for collapse(2)
      for (int k = 0; k < BT_cols; ++k) {
        for (int m = 0; m < width2; ++m) {
          float sum = 0;
#pragma omp simd reduction(+ : sum)
          for (int n = 0; n < width1; ++n) {
            sum += A.at(j + m, i + n) * BT.at(i + n, k);
          }
#pragma omp atomic
          data.at(j + m, k) += sum;
        }
      }
    }
  }

  return data;
}

