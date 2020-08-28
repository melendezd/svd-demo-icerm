from __future__ import annotations

from nptyping import NDArray, UInt8, Float64  # type: ignore
from typing import Any, Callable, TypeVar, Generic, Optional

import numpy as np  # type: ignore
import numpy.linalg as la  # type: ignore

import streamlit as st  #type: ignore

@st.cache
def svd(matrix: NDArray[(Any, Any), Any]) -> Any:
    return np.linalg.svd(matrix.astype(np.float64), full_matrices=False)


'''
Computes the randomized singular value decomposition of the input matrix.
'''
def randomized_svd(matrix: NDArray[(Any, Any), Any], rank: int, oversample: int=10,
                   power_iterations: int = 0, full_matrices: bool = False)\
                   -> Any:
    matrix = matrix.astype(np.float64)
    rows, columns = matrix.shape

    # Create a random projection matrix
    projector = np.random.rand(columns, rank + oversample)

    # Sample from the column space of X
    sample = matrix @ projector

    # Perform power iteration
    for i in range(0, power_iterations):
        sample = matrix @ (matrix.T @ sample)

    orthogonal, r = np.linalg.qr(sample)

    # Project X into the sampled subspace
    projected = orthogonal.T @ matrix

    # Obtain the SVD for this smaller matrix and recover the SVD for matrix
    u_projected, s, v = np.linalg.svd(projected, full_matrices)
    u = orthogonal @ u_projected

    return u, s, v


'''
Computes the compressed singular value decomposition of the input matrix.
'''
def compressed_svd(matrix: NDArray[(Any, Any), Any], rank: int, oversample:int=5,
                   density:float=3) -> Any:
    rows, cols = matrix.shape
    matrix = matrix.astype(np.float64)

    if density == None:
        test_matrix = np.random.rand(rank + oversample, rows)
    else:
        # Sparse random test matrix
        test_matrix = np.random.choice(a=[1, 0, -1], 
                p=[1/(2*density), 1 - 1/density, 1/(2*density)],
                size=(rank + oversample, rows))

    # Y
    sketch = test_matrix @ matrix

    # Outer product of sketch matrix with itself to obtain right singular vectors
    # B
    outer = sketch @ sketch.T

    # Ensure symmetry
    #outer = 1/2 * (outer + outer.T) 

    #T, V
    outer_eigenvalues, outer_eigenvectors = la.eigh(outer)
    outer_eigenvectors_trunc = np.flip(outer_eigenvectors[:, -rank:],axis=1)
    outer_eigenvalues_trunc = np.flip(outer_eigenvalues[-rank:])
    print(outer_eigenvalues)
    print(outer_eigenvalues_trunc)

    singular_values = np.sqrt(outer_eigenvalues_trunc)
    singular_values_matrix = np.diag(singular_values)

    right_singular_vectors = sketch.T @ outer_eigenvectors_trunc @ la.inv(singular_values_matrix)
    scaled_left_singular_vectors = matrix @ right_singular_vectors

    left_singular_vectors_updated, singular_values_updated, right_sv_multiplier = la.svd(scaled_left_singular_vectors, full_matrices=False)

    right_singular_vectors_updated = right_singular_vectors @ right_sv_multiplier.T

    return left_singular_vectors_updated, singular_values_updated, right_singular_vectors_updated.T


'''
Computes a rank <rank> approximation of a matrix with optional oversampling and randomization 
'''
def rank_k_approx(matrix: NDArray[(Any, Any), Any], rank:int=None,
                  mode='deterministic', oversample:int=10,
                  power_iterations=0) -> NDArray[(Any, Any), Float64]:

    matrix = matrix.astype(np.float64)
    mode_letter = mode[0]

    if mode_letter == 'd':
        u, s, vh = svd(matrix)
    elif mode_letter == 'r':
        if rank is not None:
            u, s, vh = randomized_svd(matrix, rank, oversample, power_iterations)
        else:
            raise Exception("Must provide rank to randomized SVD")
    elif mode_letter == 'c':
        if rank == None:
            raise Exception("Must provide rank to compressed SVD")
        if rank is not None:
            u, s, vh = compressed_svd(matrix, rank, oversample)
        else:
            raise Exception("Must provide rank to randomized SVD")

    return u[:, :rank] @ np.diag(s[:rank]) @ vh[:rank, :]
