import numpy as np
import sympy as sp

def canonicalize(i, j, k, l):
    """
    Return the canonical representative of the orbit {(i,j,k,l), (j,i,l,k)}
    using lexicographic order.
    """
    a = (i, j, k, l)
    b = (j, i, l, k)
    return a if a <= b else b

def build_canonical_map(d=3):
    """
    Build:
      - canon_list: list of canonical 4-tuples in lex order
      - canon_index: dict mapping canonical 4-tuple -> compact index [0..N-1]
      - fetch_idx(i,j,k,l): returns compact index for any indices via canonicalization
    """
    seen = set()
    canon_list = []
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    c = canonicalize(i, j, k, l)
                    if c not in seen:
                        seen.add(c)
                        canon_list.append(c)
    canon_list.sort()  # stable, deterministic order
    canon_index = {t: idx for idx, t in enumerate(canon_list)}

    def fetch_idx(i, j, k, l):
        return canon_index[canonicalize(i, j, k, l)]

    return canon_list, canon_index, fetch_idx

def pack_tensor(S, d=3, debug=False):
    """
    Pack a rank-4 tensor S[i,j,k,l] (numpy or sympy) into a minimal vector
    using only the symmetry S_{ijkl} = S_{jilk}.
    """
    canon_list, canon_index, _ = build_canonical_map(d)
    N = len(canon_list)
    vec = [None]*N
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    idx = canon_index[canonicalize(i,j,k,l)]
                    val = S[i,j,k,l]
                    if vec[idx] is None:
                        vec[idx] = val
                    elif debug:
                        # Optional consistency check: values mapping to same orbit must match
                        if hasattr(val, 'equals'):  # sympy
                            assert sp.simplify(val - vec[idx]) == 0
                        else:
                            assert np.allclose(val, vec[idx])
    # Replace any remaining None with zeros (shouldn't happen if S is fully filled)
    vec = [0 if v is None else v for v in vec]
    return vec, canon_list  # keep canon_list for reconstruction

def unpack_get(vec, canon_list, i, j, k, l):
    """
    Random access getter: retrieve S[i,j,k,l] from packed vector.
    """
    c = canonicalize(i, j, k, l)
    return vec[canon_list.index(c)]  # small; for speed, prebuild a dict

def build_fast_getter(vec, canon_list):
    """
    Build a fast O(1) getter using a dict index (recommended).
    """
    index = {t: idx for idx, t in enumerate(canon_list)}
    def get(i, j, k, l):
        return vec[index[canonicalize(i, j, k, l)]]
    return get

def reconstruct_full(vec, canon_list, d=3, as_sympy=False):
    """
    Reconstruct full S[i,j,k,l] array obeying S_{ijkl}=S_{jilk}.
    """
    S = sp.MutableDenseNDimArray.zeros(d,d,d,d) if as_sympy else np.zeros((d,d,d,d))
    idx = {t: p for p, t in enumerate(canon_list)}
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    c = canonicalize(i,j,k,l)
                    S[i,j,k,l] = vec[idx[c]]
    return S