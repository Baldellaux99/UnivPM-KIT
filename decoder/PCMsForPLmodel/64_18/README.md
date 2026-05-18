# PCM files for PL model

This folder contains the saved matrices for the quantum CSS code with parameters:

```text
n = 64
k = 18
m = 64
```

The files are saved in:

```text
./PCMsForPLmodel/64_18/
```

## Files

```text
64_18_64_H.txt
64_18_G.txt
64_18_Rx.txt
64_18_Rz.txt
```

## Source matrices

The original CSS parity-check matrices are:

```text
H_X shape = (32, 64)
H_Z shape = (32, 64)
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
rank(H_X) = 23
rank(H_Z) = 23
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
64_18_64_H.txt
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
64_18_G.txt
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
G_X shape = (41, 64)
G_Z shape = (41, 64)
```

The file is saved as a full dense matrix, not alist.

Each row is written as space-separated integers.

## R matrices

The files:

```text
64_18_Rx.txt
64_18_Rz.txt
```

store binary matrices in the same alist-with-values format as `H`.

They satisfy:

```text
Rx * H_X1 = H_X2    over GF(2)
Rz * H_Z1 = H_Z2    over GF(2)
```

with shapes:

```text
Rx shape = (9, 23)
Rz shape = (9, 23)
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
