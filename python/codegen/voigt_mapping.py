#!/usr/bin/env python3

import sympy as sp
import numpy as np

def voigt_mapping(d=3):
    """
    Return a mapping from symmetric index pairs (i,j) to Voigt index I
    for dimension d.
    """
    mapping = {}
    counter = 0
    for i in range(d):
        for j in range(i, d):  # symmetric pairs
            mapping[(i, j)] = counter
            mapping[(j, i)] = counter
            counter += 1
    return mapping

def tensor4_to_voigt(S, d=3):
    """
    Convert a 4th-order symmetric tensor S_{ijkl} with full symmetry
    into a symmetric Voigt matrix representation C_{IJ}.
    """
    mapping = voigt_mapping(d)
    m = d*(d+1)//2
    C = sp.MutableDenseMatrix(m, m, [0]*m*m)

    for i in range(d):
        for j in range(d):
            I = mapping[(i, j)]
            for k in range(d):
                for l in range(d):
                    J = mapping[(k, l)]
                    C[I, J] = S[i, j, k, l]
    return sp.Matrix(C)

def extract_independent(C):
    """
    Extract independent components (upper triangular) of a symmetric Voigt matrix.
    """
    indep = []
    for i in range(C.shape[0]):
        for j in range(i, C.shape[1]):
            indep.append(C[i,j])
    return indep

def independent_to_voigt(indep, d=3):
    """
    Reconstruct Voigt symmetric matrix from list of independent components.
    """
    m = d*(d+1)//2
    C = sp.MutableDenseMatrix(m, m, [0]*m*m)
    idx = 0
    for i in range(m):
        for j in range(i, m):
            C[i,j] = indep[idx]
            C[j,i] = indep[idx]  # symmetry
            idx += 1
    return sp.Matrix(C)

def voigt_to_tensor4(C, d=3):
    """
    Convert a Voigt matrix back to a 4th-order tensor S_{ijkl}.
    """
    mapping = voigt_mapping(d)
    S = sp.MutableDenseNDimArray.zeros(d, d, d, d)

    for i in range(d):
        for j in range(d):
            I = mapping[(i,j)]
            for k in range(d):
                for l in range(d):
                    J = mapping[(k,l)]
                    S[i,j,k,l] = C[I,J]
    return S



if __name__ == "__main__":
    # ------------------------
    # Example usage in 3D
    # ------------------------
    d = 3
    # Create a symbolic 4th-order tensor (81 entries, redundant)
    ijkl = sp.MutableDenseNDimArray.zeros(d, d, d, d)
    symbols = sp.symbols('C0:81')  
    for n, idx in enumerate(np.ndindex(d, d, d, d)):
        ijkl[idx] = symbols[n]

    # (1) Tensor → Voigt
    C_voigt = tensor4_to_voigt(ijkl, d)

    # (2) Extract 21 unique entries
    indep_entries = extract_independent(C_voigt)

    # (3) Reconstruct Voigt from 21 entries
    C_reconstructed = independent_to_voigt(indep_entries, d)

    # (4) Reconstruct full tensor
    ijkl_reconstructed = voigt_to_tensor4(C_reconstructed, d)

    print("Number of independent entries:", len(indep_entries))
    print("Independent entries:", indep_entries[:])  # show first 5