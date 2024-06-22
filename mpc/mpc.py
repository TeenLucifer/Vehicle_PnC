import math
import osqp
import numpy as np
from scipy import sparse
from qpsolvers import solve_qp

class MPC:
    def __init__(self, Ad, Bd, Q, R, Qf, N = 10):
        self.Ad = Ad
        self.Bd = Bd
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.N = N    # 预测步数
        self.nx = Bd.shape[0]
        self.nu = Bd.shape[1]

    def solve(self, x0, Ad, Bd, Q, R, Qf, N = 10):
        self.Ad = Ad
        self.Bd = Bd
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.N = N    # 预测步数
        self.nx = Bd.shape[0]
        self.nu = Bd.shape[1]

        A_powers = []
        for i in range(self.N + 1):
            A_powers.append(np.linalg.matrix_power(Ad, i))

        C = np.zeros(((self.N + 1) * self.nx, self.N * self.nu))
        M = np.zeros(((self.N + 1) * self.nx, self.nx))
        for i in range(self.N + 1):
            for j in range(self.N):
                if i - j - 1 >= 0:
                    C_ij = A_powers[i - j - 1] * self.Bd
                    C[i * self.nx : (i + 1) * self.nx, j * self.nu : (j + 1) * self.nu] = C_ij
                else:
                    C_ij = np.zeros((self.nx, self.nu))
                    C[i * self.nx : (i + 1) * self.nx, j * self.nu : (j + 1) * self.nu] = C_ij
            M[i * self.nx : (i + 1) * self.nx, :] = A_powers[i]

        Q_bar = np.kron(np.eye(self.N + 1), Q)
        Q_bar[self.N * self.nx : (1 + self.N) * self.nx, self.N * self.nx : (1 + self.N) * self.nx:] = Qf
        R_bar = np.kron(np.eye(self.N), R)
        E = M.T * Q_bar * C

        P = 2 * C.T * Q_bar * C + R_bar
        q = 2 * E.T * x0

        # Gx <= h
        G_ = np.eye(self.N * self.nu)
        G = np.block([                   # 不等式约束矩阵
            [G_, np.zeros_like(G_)],
            [np.zeros_like(G_), -G_]
        ])
        h = np.vstack(np.ones((2 * self.N * self.nu, 1)) * 999) # 不等式约束向量

        # Ax = b
        A = None # 等式约束矩阵
        b = None # 等式约束向量

        # 转换为稀疏矩阵的形式能加速计算
        P = sparse.csc_matrix(P)
        q = np.asarray(q)
        if G is None:
            pass
        else:
            G = sparse.csc_matrix(G)
        if A is None:
            pass
        else:
            A = sparse.csc_matrix(A)

        res = solve_qp(P, q, G, h, A, b, solver="osqp")

        return res