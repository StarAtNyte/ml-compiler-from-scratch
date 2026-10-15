"""CSR and scatter-add semantics, including duplicate indices.

Run: python -m labs.irregular. CPU reference algorithms, not optimized kernels.
"""
import numpy as np


def csr_matvec(rowptr, columns, values, vector):
  rowptr, columns, values, vector = map(np.asarray, (rowptr, columns, values, vector))
  if any(x.ndim != 1 for x in (rowptr, columns, values, vector)) or len(rowptr) < 1:
    raise ValueError("expected one-dimensional CSR arrays and vector")
  if not all(np.issubdtype(x.dtype, np.integer) for x in (rowptr, columns)):
    raise ValueError("offsets and indices must be integer")
  if rowptr[0] != 0 or rowptr[-1] != len(values) or len(columns) != len(values) or np.any(np.diff(rowptr) < 0):
    raise ValueError("invalid CSR offsets")
  if np.any(columns < 0) or np.any(columns >= len(vector)): raise ValueError("column out of bounds")
  result = np.zeros(len(rowptr)-1, dtype=np.result_type(values, vector))
  for row in range(len(result)):
    start, end = rowptr[row:row+2]
    result[row] = np.sum(values[start:end] * vector[columns[start:end]])
  return result


def embedding_gradient(indices, grad_output, vocabulary_size):
  indices, grad_output = np.asarray(indices), np.asarray(grad_output)
  if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or grad_output.ndim != 2 or len(indices) != len(grad_output):
    raise ValueError("invalid indices or output gradient")
  if vocabulary_size < 1 or np.any(indices < 0) or np.any(indices >= vocabulary_size):
    raise ValueError("embedding index out of bounds")
  result = np.zeros((vocabulary_size, grad_output.shape[1]), dtype=grad_output.dtype)
  # result[indices] += grad_output would lose repeated-index updates.
  np.add.at(result, indices, grad_output)
  return result


def main():
  print("CSR with an empty row:", csr_matvec([0, 2, 2, 3], [0, 2, 1], [2., 3., 4.], [1., 2., 3.]))
  print("repeated-token gradient:", embedding_gradient([1, 1, 2], np.ones((3, 2)), 4).tolist())


if __name__ == "__main__": main()
