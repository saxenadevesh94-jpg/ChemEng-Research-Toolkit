import numpy as np
import pytest

from src.cfd import LinearSystem, SparseMatrix


# ---------------------------------------------------------------------------
# SparseMatrix — creation
# ---------------------------------------------------------------------------

def test_sparse_matrix_stores_size():
    mat = SparseMatrix(size=4)
    assert mat.size == 4


def test_sparse_matrix_size_must_be_int():
    with pytest.raises(TypeError):
        SparseMatrix(size=3.0)


def test_sparse_matrix_size_must_be_positive():
    with pytest.raises(ValueError):
        SparseMatrix(size=0)

    with pytest.raises(ValueError):
        SparseMatrix(size=-1)


def test_sparse_matrix_starts_empty():
    mat = SparseMatrix(size=5)
    assert mat.get(0, 0) == 0.0
    assert mat.get(2, 3) == 0.0


# ---------------------------------------------------------------------------
# SparseMatrix — coefficient assignment and retrieval
# ---------------------------------------------------------------------------

def test_set_and_get_coefficient():
    mat = SparseMatrix(size=3)
    mat.set(1, 2, 7.5)
    assert mat.get(1, 2) == 7.5


def test_set_overwrites_previous_value():
    mat = SparseMatrix(size=3)
    mat.set(0, 0, 1.0)
    mat.set(0, 0, 99.0)
    assert mat.get(0, 0) == 99.0


def test_set_diagonal_and_off_diagonal():
    mat = SparseMatrix(size=4)
    mat.set(0, 0, 4.0)
    mat.set(0, 1, -1.0)
    mat.set(1, 0, -1.0)
    mat.set(1, 1, 4.0)
    assert mat.get(0, 0) == 4.0
    assert mat.get(0, 1) == -1.0
    assert mat.get(1, 0) == -1.0
    assert mat.get(2, 3) == 0.0


# ---------------------------------------------------------------------------
# SparseMatrix — invalid indices
# ---------------------------------------------------------------------------

def test_set_row_out_of_range_raises_error():
    mat = SparseMatrix(size=3)
    with pytest.raises(ValueError):
        mat.set(3, 0, 1.0)

    with pytest.raises(ValueError):
        mat.set(-1, 0, 1.0)


def test_set_col_out_of_range_raises_error():
    mat = SparseMatrix(size=3)
    with pytest.raises(ValueError):
        mat.set(0, 3, 1.0)


def test_get_row_out_of_range_raises_error():
    mat = SparseMatrix(size=3)
    with pytest.raises(ValueError):
        mat.get(5, 0)


def test_get_col_out_of_range_raises_error():
    mat = SparseMatrix(size=3)
    with pytest.raises(ValueError):
        mat.get(0, 5)


def test_non_integer_indices_raise_type_error():
    mat = SparseMatrix(size=3)
    with pytest.raises(TypeError):
        mat.set(0.0, 1, 1.0)

    with pytest.raises(TypeError):
        mat.get(1, 1.5)


# ---------------------------------------------------------------------------
# SparseMatrix — to_array
# ---------------------------------------------------------------------------

def test_to_array_returns_correct_dense_matrix():
    mat = SparseMatrix(size=3)
    mat.set(0, 0, 2.0)
    mat.set(0, 1, -1.0)
    mat.set(1, 0, -1.0)
    mat.set(1, 1, 2.0)
    mat.set(2, 2, 1.0)

    expected = np.array([
        [ 2., -1.,  0.],
        [-1.,  2.,  0.],
        [ 0.,  0.,  1.],
    ])
    assert np.allclose(mat.to_array(), expected)


def test_to_array_shape_matches_size():
    mat = SparseMatrix(size=5)
    assert mat.to_array().shape == (5, 5)


def test_to_array_all_zeros_when_empty():
    mat = SparseMatrix(size=3)
    assert np.allclose(mat.to_array(), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# LinearSystem — creation
# ---------------------------------------------------------------------------

def test_linear_system_stores_matrix_and_rhs():
    mat = SparseMatrix(size=3)
    b = np.array([1.0, 2.0, 3.0])
    system = LinearSystem(mat, b)
    assert system.matrix is mat
    assert np.allclose(system.rhs, b)


def test_linear_system_size_property():
    mat = SparseMatrix(size=4)
    b = np.zeros(4)
    system = LinearSystem(mat, b)
    assert system.size == 4


def test_linear_system_accepts_list_as_rhs():
    mat = SparseMatrix(size=2)
    system = LinearSystem(mat, [0.0, 1.0])
    assert system.rhs.dtype == float


# ---------------------------------------------------------------------------
# LinearSystem — dimension mismatch
# ---------------------------------------------------------------------------

def test_rhs_length_mismatch_raises_error():
    mat = SparseMatrix(size=3)
    with pytest.raises(ValueError):
        LinearSystem(mat, np.array([1.0, 2.0]))


def test_rhs_2d_array_raises_error():
    mat = SparseMatrix(size=2)
    with pytest.raises(ValueError):
        LinearSystem(mat, np.ones((2, 2)))


def test_matrix_must_be_sparse_matrix():
    with pytest.raises(TypeError):
        LinearSystem(np.eye(3), np.zeros(3))


# ---------------------------------------------------------------------------
# LinearSystem — summary
# ---------------------------------------------------------------------------

def test_summary_contains_size():
    mat = SparseMatrix(size=5)
    b = np.zeros(5)
    system = LinearSystem(mat, b)
    assert "5" in system.summary()


def test_summary_contains_nnz_count():
    mat = SparseMatrix(size=3)
    mat.set(0, 0, 1.0)
    mat.set(1, 1, 2.0)
    b = np.zeros(3)
    system = LinearSystem(mat, b)
    # Two coefficients were stored.
    assert "2" in system.summary()


def test_summary_is_a_string():
    mat = SparseMatrix(size=2)
    system = LinearSystem(mat, np.zeros(2))
    assert isinstance(system.summary(), str)


def test_summary_empty_matrix_shows_zero_nnz():
    mat = SparseMatrix(size=4)
    system = LinearSystem(mat, np.zeros(4))
    assert "0" in system.summary()
