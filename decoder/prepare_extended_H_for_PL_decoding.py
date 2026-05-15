import numpy as np
import pandas as pd
import galois

GF2 = galois.GF(2)

from pathlib import Path


def to_numpy_uint8(A):
    """
    Convert a galois array or numpy array to ordinary uint8 numpy array.
    """
    return np.array(A, dtype=np.uint8)


def build_quaternary_H(Hx, Hz):
    """
    Build quaternary H in the form:

        H = [ 1 * Hx ]
            [ 2 * Hz ]

    Entries are stored as integers 0, 1, 2.
    """
    Hx = to_numpy_uint8(Hx) % 2
    Hz = to_numpy_uint8(Hz) % 2

    if Hx.shape[1] != Hz.shape[1]:
        raise ValueError("Hx and Hz must have the same number of columns.")

    H_quat = np.vstack([
        Hx,
        2 * Hz,
    ])

    return H_quat.astype(np.uint8)


def build_quaternary_G(Gx, Gz):
    """
    Build quaternary G in the form:

        G = [ 1 * Gx ]
            [ 2 * Gz ]

    Entries are stored as integers 0, 1, 2.
    """
    Gx = to_numpy_uint8(Gx) % 2
    Gz = to_numpy_uint8(Gz) % 2

    if Gx.shape[1] != Gz.shape[1]:
        raise ValueError("Gx and Gz must have the same number of columns.")

    G_quat = np.vstack([
        Gx,
        2 * Gz,
    ])

    return G_quat.astype(np.uint8)


def write_full_matrix_txt(A, path):
    """
    Save a full matrix as plain text with space-separated entries.
    """
    A = to_numpy_uint8(A)

    with open(path, "w") as f:
        for row in A:
            f.write(" ".join(map(str, row)) + "\n")


def write_nonbinary_alist(A, path):
    """
    Save a nonbinary/quaternary matrix in alist format.

    Matrix A has shape:

        m x n

    The file format is:

        n m
        max_col_weight max_row_weight
        column_weights
        row_weights
        column_neighbor_indices
        row_neighbor_indices
        row_neighbor_values
        column_neighbor_values

    Indices are 1-based and padded with 0.
    Values are padded with 0.
    """
    A = to_numpy_uint8(A)

    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")

    m, n = A.shape

    row_weights = np.count_nonzero(A, axis=1)
    col_weights = np.count_nonzero(A, axis=0)

    max_row_weight = int(row_weights.max()) if m > 0 else 0
    max_col_weight = int(col_weights.max()) if n > 0 else 0

    col_indices = []
    col_values = []

    for j in range(n):
        rows = np.flatnonzero(A[:, j]) + 1  # 1-based
        vals = A[rows - 1, j]

        rows = list(map(int, rows))
        vals = list(map(int, vals))

        rows += [0] * (max_col_weight - len(rows))
        vals += [0] * (max_col_weight - len(vals))

        col_indices.append(rows)
        col_values.append(vals)

    row_indices = []
    row_values = []

    for i in range(m):
        cols = np.flatnonzero(A[i, :]) + 1  # 1-based
        vals = A[i, cols - 1]

        cols = list(map(int, cols))
        vals = list(map(int, vals))

        cols += [0] * (max_row_weight - len(cols))
        vals += [0] * (max_row_weight - len(vals))

        row_indices.append(cols)
        row_values.append(vals)

    with open(path, "w") as f:
        f.write(f"{n} {m}\n")
        f.write(f"{max_col_weight} {max_row_weight}\n")

        f.write(" ".join(map(str, map(int, col_weights))) + "\n")
        f.write(" ".join(map(str, map(int, row_weights))) + "\n")

        for row in col_indices:
            f.write(" ".join(map(str, row)) + "\n")

        for row in row_indices:
            f.write(" ".join(map(str, row)) + "\n")

        for row in row_values:
            f.write(" ".join(map(str, row)) + "\n")

        for row in col_values:
            f.write(" ".join(map(str, row)) + "\n")


def save_for_pl_model(Hx, Hz, Gx, Gz, base_dir="../PCMsForPLmodel", k=None):
    """
    Save H and G for the PL model.

    H is saved as nonbinary alist:

        H = [ 1 * Hx ]
            [ 2 * Hz ]

    filename:

        n_k_m_H.txt

    G is saved as a full GF(4)-valued integer matrix:

        G = [ 1 * Gx ]
            [ 2 * Gz ]

    filename:

        n_k_G.txt

    Output folder:

        ../PCMsForPLmodel/n_k
    """
    Hx = as_gf2_matrix(Hx)
    Hz = as_gf2_matrix(Hz)
    Gx = as_gf2_matrix(Gx)
    Gz = as_gf2_matrix(Gz)

    n = Hx.shape[1]

    if Hz.shape[1] != n:
        raise ValueError("Hx and Hz must have the same number of columns.")

    if Gx.shape[1] != n or Gz.shape[1] != n:
        raise ValueError("Gx and Gz must have the same number of columns as Hx and Hz.")

    rank_Hx = gf2_rank(Hx)
    rank_Hz = gf2_rank(Hz)

    if k is None:
        k = n - rank_Hx - rank_Hz

    H_quat = build_quaternary_H(Hx, Hz)
    G_quat = build_quaternary_G(Gx, Gz)

    m = H_quat.shape[0]

    output_dir = Path(base_dir) / f"{n}_{k}"
    output_dir.mkdir(parents=True, exist_ok=True)

    H_path = output_dir / f"{n}_{k}_{m}_H.txt"
    G_path = output_dir / f"{n}_{k}_G.txt"

    write_nonbinary_alist(H_quat, H_path)
    write_full_matrix_txt(G_quat, G_path)

    print("Saved H to:", H_path)
    print("Saved G to:", G_path)

    print("H_quat shape:", H_quat.shape)
    print("G_quat shape:", G_quat.shape)
    print("rank(Hx):", rank_Hx)
    print("rank(Hz):", rank_Hz)
    print("k:", k)

    return {
        "H_path": H_path,
        "G_path": G_path,
        "H_quat": H_quat,
        "G_quat": G_quat,
        "n": n,
        "k": k,
        "m": m,
    }
def as_gf2_matrix(A):
    """
    Convert an array-like object to a GF(2) matrix.
    """
    A = np.asarray(A, dtype=np.uint8) % 2

    if A.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")

    return GF2(A)


def load_binary_csv(path):
    """
    Load a comma-separated binary matrix as a GF(2) matrix.
    """
    A = pd.read_csv(path, sep=",", header=None, dtype=np.uint8).to_numpy()
    return as_gf2_matrix(A)


def gf2_rank(A):
    """
    Compute rank over GF(2).
    """
    A = as_gf2_matrix(A)
    return A.row_space().shape[0]


def pivot_columns_from_rref(A_rref):
    """
    Given an RREF matrix, return the pivot column indices.
    """
    pivots = []

    for row in A_rref:
        nonzero_cols = np.flatnonzero(np.array(row, dtype=np.uint8))

        if len(nonzero_cols) > 0:
            pivots.append(int(nonzero_cols[0]))

    return pivots


def find_independent_row_indices(H):
    """
    Find rank(H) linearly independent row indices of H over GF(2).

    This uses the fact that independent rows of H correspond to
    pivot columns of H.T.
    """
    H = as_gf2_matrix(H)

    H_T_rref = H.T.row_reduce()
    independent_rows = pivot_columns_from_rref(H_T_rref)

    return independent_rows


def solve_R_for_H2_equals_RH1(H1, H2):
    """
    Solve for R over GF(2) such that:

        R @ H1 = H2

    where the rows of H1 are linearly independent and span the rows of H2.
    """
    H1 = as_gf2_matrix(H1)
    H2 = as_gf2_matrix(H2)

    r, n = H1.shape
    num_H2_rows = H2.shape[0]

    if num_H2_rows == 0:
        return GF2.Zeros((0, r))

    if r == 0:
        if np.any(H2 != 0):
            raise ValueError("No R exists because H1 has rank 0 but H2 is nonzero.")
        return GF2.Zeros((num_H2_rows, 0))

    # We want:
    #
    #     R @ H1 = H2
    #
    # Transpose:
    #
    #     H1.T @ R.T = H2.T
    #
    # Let:
    #
    #     A = H1.T
    #     X = R.T
    #     B = H2.T
    #
    # Solve:
    #
    #     A @ X = B
    #
    A = H1.T
    B = H2.T

    augmented = GF2(np.hstack([A, B]))
    augmented_rref = augmented.row_reduce(ncols=r)

    left = augmented_rref[:, :r]
    right = augmented_rref[:, r:]

    I = GF2.Identity(r)
    Z = GF2.Zeros((n - r, r))

    if not np.array_equal(left[:r, :], I):
        raise RuntimeError("Unexpected RREF result: top-left block is not identity.")

    if not np.array_equal(left[r:, :], Z):
        raise RuntimeError("Unexpected RREF result: lower-left block is not zero.")

    if np.any(right[r:, :] != 0):
        raise ValueError("No R exists: some rows of H2 are not in the row span of H1.")

    X = right[:r, :]   # X = R.T
    R = X.T

    if not np.array_equal(R @ H1, H2):
        raise RuntimeError("Computed R does not satisfy R @ H1 == H2.")

    return R


def decompose_H(H):
    """
    Find rank(H) independent rows of H as H1, permute H into

        H_perm = [H1]
                 [H2]

    and compute R such that

        R @ H1 = H2

    all over GF(2).
    """
    H = as_gf2_matrix(H)

    m, n = H.shape

    independent_rows = find_independent_row_indices(H)
    independent_set = set(independent_rows)

    dependent_rows = [i for i in range(m) if i not in independent_set]

    perm = independent_rows + dependent_rows

    H_perm = H[perm, :]

    r = len(independent_rows)

    H1 = H_perm[:r, :]
    H2 = H_perm[r:, :]

    R = solve_R_for_H2_equals_RH1(H1, H2)

    return H_perm, H1, H2, R, independent_rows, perm


def make_dual_matrix(H):
    """
    Compute a basis for the nullspace of H over GF(2).

    The returned matrix G has rows satisfying:

        H @ G.T = 0
    """
    H = as_gf2_matrix(H)

    G = H.null_space()

    if not np.all(H @ G.T == 0):
        raise RuntimeError("Dual/nullspace verification failed.")

    return G

def save_R_matrices_for_pl_model(Rx, Rz, n, k, base_dir="../PCMsForPLmodel"):
    """
    Save two binary R matrices in the same alist-with-values format as H.

    Files:
        ../PCMsForPLmodel/n_k/n_k_Rx.txt
        ../PCMsForPLmodel/n_k/n_k_Rz.txt
    """
    Rx = to_numpy_uint8(Rx) % 2
    Rz = to_numpy_uint8(Rz) % 2

    output_dir = Path(base_dir) / f"{n}_{k}"
    output_dir.mkdir(parents=True, exist_ok=True)

    Rx_path = output_dir / f"{n}_{k}_Rx.txt"
    Rz_path = output_dir / f"{n}_{k}_Rz.txt"

    write_nonbinary_alist(Rx, Rx_path)
    write_nonbinary_alist(Rz, Rz_path)

    print("Saved Rx to:", Rx_path)
    print("Saved Rz to:", Rz_path)

    print("Rx shape:", Rx.shape)
    print("Rz shape:", Rz.shape)

    return {
        "Rx_path": Rx_path,
        "Rz_path": Rz_path,
        "Rx": Rx,
        "Rz": Rz,
    }
def analyze_pcm(path, name="H"):
    """
    Load a parity-check matrix, compute its rank, nullspace/dual matrix,
    decompose it into H1/H2, and compute R such that R @ H1 = H2.
    """
    H = load_binary_csv(path)

    rank_H = gf2_rank(H)
    G_dual = make_dual_matrix(H)

    H_perm, H1, H2, R, independent_rows, perm = decompose_H(H)

    print(f"\n{name}")
    print("-" * len(name))
    print("H shape:", H.shape)
    print("rank(H) over GF(2):", rank_H)
    print("G_dual shape:", G_dual.shape)
    print("H1 shape:", H1.shape)
    print("H2 shape:", H2.shape)
    print("R shape:", R.shape)
    print("Independent row indices:", independent_rows)

    print("Check H @ G_dual.T == 0:", np.all(H @ G_dual.T == 0))
    print("Check H_perm == [H1; H2]:", np.array_equal(H_perm, H[perm, :]))
    print("Check R @ H1 == H2:", np.array_equal(R @ H1, H2))

    return {
        "H": H,
        "rank": rank_H,
        "G_dual": G_dual,
        "H_perm": H_perm,
        "H1": H1,
        "H2": H2,
        "R": R,
        "independent_rows": independent_rows,
        "perm": perm,
    }
from pathlib import Path
from textwrap import dedent


def write_pl_model_readme(save_data, HX_data, HZ_data, base_dir="./PCMsForPLmodel"):
    """
    Write a README explaining the saved matrix files and formats.
    """

    n = save_data["n"]
    k = save_data["k"]
    m = save_data["m"]

    output_dir = Path(base_dir) / f"{n}_{k}"
    output_dir.mkdir(parents=True, exist_ok=True)

    readme_path = output_dir / "README.md"

    hx_shape = HX_data["H_perm"].shape
    hz_shape = HZ_data["H_perm"].shape

    hx_rank = HX_data["H1"].shape[0]
    hz_rank = HZ_data["H1"].shape[0]

    gx_shape = HZ_data["G_dual"].shape
    gz_shape = HX_data["G_dual"].shape

    rx_shape = HX_data["R"].shape
    rz_shape = HZ_data["R"].shape

    text = f"""
    # PCM files for PL model

    This folder contains the saved matrices for the quantum CSS code with parameters:

    ```text
    n = {n}
    k = {k}
    m = {m}
    ```

    The files are saved in:

    ```text
    ./PCMsForPLmodel/{n}_{k}/
    ```

    ## Files

    ```text
    {n}_{k}_{m}_H.txt
    {n}_{k}_G.txt
    {n}_{k}_Rx.txt
    {n}_{k}_Rz.txt
    ```

    ## Source matrices

    The original CSS parity-check matrices are:

    ```text
    H_X shape = {hx_shape}
    H_Z shape = {hz_shape}
    ```

    Before saving, both matrices are permuted so that the linearly independent rows come first:

    ```text
    H_X_perm = [H_X1]
               [H_X2]

    H_Z_perm = [H_Z1]
               [H_Z2]
    ```

    where:

    ```text
    rank(H_X) = {hx_rank}
    rank(H_Z) = {hz_rank}
    ```

    Therefore:

    ```text
    H_X1 = first rank(H_X) rows of H_X_perm
    H_X2 = remaining rows of H_X_perm

    H_Z1 = first rank(H_Z) rows of H_Z_perm
    H_Z2 = remaining rows of H_Z_perm
    ```

    ## H matrix

    The file:

    ```text
    {n}_{k}_{m}_H.txt
    ```

    stores the quaternary parity-check matrix

    ```text
    H = [ 1 * H_X_perm ]
        [ 2 * H_Z_perm ]
    ```

    The entries are represented as integers:

    ```text
    0 = zero
    1 = X-type binary entry
    2 = Z-type binary entry
    ```

    Thus, rows coming from `H_X_perm` contain only values `0` and `1`, while rows coming from `H_Z_perm` contain only values `0` and `2`.

    This file is saved in nonbinary alist format with element values.

    ## G matrix

    The file:

    ```text
    {n}_{k}_G.txt
    ```

    stores the full quaternary matrix

    ```text
    G = [ 1 * G_X ]
        [ 2 * G_Z ]
    ```

    where:

    ```text
    G_X = nullspace(H_Z)
    G_Z = nullspace(H_X)
    ```

    with shapes:

    ```text
    G_X shape = {gx_shape}
    G_Z shape = {gz_shape}
    ```

    The file is saved as a full dense matrix, not alist.

    Each row is written as space-separated integers.

    ## R matrices

    The files:

    ```text
    {n}_{k}_Rx.txt
    {n}_{k}_Rz.txt
    ```

    store binary matrices in the same alist-with-values format as `H`.

    They satisfy:

    ```text
    Rx * H_X1 = H_X2    over GF(2)
    Rz * H_Z1 = H_Z2    over GF(2)
    ```

    with shapes:

    ```text
    Rx shape = {rx_shape}
    Rz shape = {rz_shape}
    ```

    Since `Rx` and `Rz` are binary matrices, all nonzero element values in their alist files are `1`.

    ## Alist-with-values format

    The alist files in this folder use the following structure.

    ```text
    n m
    max_col_weight max_row_weight
    column_weights
    row_weights
    column_neighbor_indices
    row_neighbor_indices
    row_neighbor_values
    column_neighbor_values
    ```

    More explicitly:

    1. First line:

    ```text
    n m
    ```

    where `n` is the number of columns and `m` is the number of rows.

    2. Second line:

    ```text
    max_col_weight max_row_weight
    ```

    3. Third line: the weight of each column.

    4. Fourth line: the weight of each row.

    5. Next `n` lines: for each column, the 1-based row indices of its nonzero entries, padded with `0`.

    6. Next `m` lines: for each row, the 1-based column indices of its nonzero entries, padded with `0`.

    7. Next `m` lines: for each row, the values of its nonzero entries, padded with `0`.

    8. Next `n` lines: for each column, the values of its nonzero entries, padded with `0`.

    All indices are 1-based. Padding entries are `0`.

    ## Notes

    All binary linear algebra is performed over GF(2).

    The quaternary matrices are stored using integer labels `0`, `1`, and `2`; no finite-field arithmetic over GF(4) is performed during saving.
    """

    text = dedent(text).strip() + "\n"

    with open(readme_path, "w") as f:
        f.write(text)

    print("Saved README to:", readme_path)

    return readme_path

if __name__ == "__main__":
    HX_PATH = r"../PCMs/QD-LDPC CSS [[128,36,8]]/QD[[128,36,8]]_H_X.csv"
    HZ_PATH = r"../PCMs/QD-LDPC CSS [[128,36,8]]/QD[[128,36,8]]_H_Z.csv"

    HX_data = analyze_pcm(HX_PATH, name="H_X")
    HZ_data = analyze_pcm(HZ_PATH, name="H_Z")

    Hx = HX_data["H_perm"]
    Hz = HZ_data["H_perm"]

    # For CSS codes:
    # Gz is the nullspace of Hx
    # Gx is the nullspace of Hz
    Gz = HX_data["G_dual"]
    Gx = HZ_data["G_dual"]

    save_data = save_for_pl_model(
        Hx=Hx,
        Hz=Hz,
        Gx=Gx,
        Gz=Gz,
        base_dir="./PCMsForPLmodel",
    )

    # R for H_X and H_Z decompositions
    Rx = HX_data["R"]
    Rz = HZ_data["R"]

    save_R_data = save_R_matrices_for_pl_model(
        Rx=Rx,
        Rz=Rz,
        n=save_data["n"],
        k=save_data["k"],
        base_dir="./PCMsForPLmodel",
    )

    readme_path = write_pl_model_readme(
        save_data=save_data,
        HX_data=HX_data,
        HZ_data=HZ_data,
        base_dir="./PCMsForPLmodel",
    )