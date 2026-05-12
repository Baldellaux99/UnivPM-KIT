import galois
import numpy as np
from itertools import product

from scipy.io import savemat

import matplotlib.pyplot as plt

import pandas as pd

gf2 = galois.GF(2)


def get_colblock2(colblock1, camel_block):
    camel_rank = np.linalg.matrix_rank(camel_block)
    all_combinations = list(product([0, 1], repeat=camel_rank))

    colblock2 = gf2.Zeros(colblock1.shape)

    for comb in all_combinations:
        col = gf2.Zeros(colblock1.shape[0])
        for i in range(camel_rank):
            if comb[i] == 1:
                col += colblock1[:, i]

        ### for each col check if it matches any col in camel_block-> if yes; comb gives you with cols of colblock1 to add to get that col and thereby form rows of colblock2
        # iterate over columns
        for j in range(camel_block.shape[1]):
            if np.array_equal(col, camel_block[:, j]):
                # ("Found matching column for column ",j
                colblock2[j, :] = gf2(list(comb))

    return colblock2


Hxprime = gf2(
    pd.read_csv("high_rank_CAMEL_matrix_F32.csv", sep=",", header=None).values
)

camel_block = Hxprime @ Hxprime.T

camel_rank = np.linalg.matrix_rank(camel_block)
print("Camel block rank:", camel_rank)
plt.spy(camel_block)
plt.show()

colblock1 = (camel_block.T.row_reduce()).T  # does col reduce
colblock1 = colblock1[:, 0:camel_rank]  # take the lin indep cols
plt.spy(colblock1)
plt.show()

colblock2 = get_colblock2(colblock1, camel_block)

test_camel_block = colblock1 @ colblock2.T

try:
    assert np.array_equal(test_camel_block, camel_block)
    print("✓ Factorization successful")
    print(colblock1 @ colblock2.T)
except AssertionError:
    raise ValueError(
        f"Factorization failed: colblock1 @ colblock2.T does not equal camel_block"
    )


H_x = np.hstack((Hxprime, colblock1))
H_z = np.hstack((Hxprime, colblock2))

n = H_x.shape[1]
k1 = n - np.linalg.matrix_rank(H_x)
k2 = n - np.linalg.matrix_rank(H_z)

k = k1 + k2 - n  # = n-rank Hx + n-rank Hz - n = n - rank Hx - rank Hz
assert k == n - np.linalg.matrix_rank(H_x) - np.linalg.matrix_rank(H_z)

##assert CSS code condition
assert np.all(gf2(H_x) @ gf2(H_z).T == 0)
print("CSS code parameters [[", n, ",", k, "]]")


Gx = H_z.null_space()
Gz = H_x.null_space()

G = np.vstack((np.array(Gx), 2 * np.array(Gz)))

H = np.vstack((np.array(H_x), 2 * np.array(H_z)))

savemat(
    "dyadic_camel_matrices" + str(n) + "_" + str(k) + ".mat",
    {"G": G, "Gx": Gx, "Gz": Gz, "H": H, "Hx": H_x, "Hz": H_z},
)
