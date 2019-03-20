import numpy as np


def v_multi(V1, V2):
    if V1.size != V2.size:
        print("wrong vectors")
        return 0
    TEMP = 0
    for (E1, E2) in zip(V1, V2):
        TEMP += E1*E2
    return TEMP


def transpose(MATRIX):
    row, column = MATRIX.shape
    MX = np.zeros((column, row))
    for i in range(0, row):
        for j in range(0, column):
            MX[j, i] = MATRIX[i, j]
    return MX


def mx_multi(MX_A, MX_B):
    A_ROW, A_COL = MX_A.shape
    B_ROW, B_COL = MX_B.shape
    MX = np.zeros([A_ROW, B_COL])

    if (A_COL != B_ROW):
        print("Cant multi given matrix")
        return MX

    for i in range(0, A_ROW):
        for j in range(0, B_COL):
            MX[i, j] = v_multi(MX_A[i, :], MX_B[:, j])
    return MX


mx = np.array([[i for i in range(0, 10)], [i for i in range(10, 20)]])
# mx = np.matrix([[1, 2, 3], [4, 5, 6]])
print(mx)

print(mx.shape)

print(" ----- ")

m1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
m2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

print(m1[0, :])
print(m2[:, 0])
print(v_multi(m1[0, :], m2[:, 0]))

print(" ----- ")

print(mx_multi(m2, m1))
print(mx_multi(m1, m2))

print(mx_multi(2 * np.ones((3, 3)), np.ones((3, 3))))
