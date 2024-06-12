import math
import numpy as np

class DiscreteLQR:
    def __init__(self, Ad, Bd, Q, R, epsilon = 1e-4):
        self.Ad = Ad # 离散系统矩阵
        self.Bd = Bd # 离散系统矩阵
        self.Q = Q
        self.R = R
        self.P = Q # 给个初始值Q
        self.K = np.mat(np.zeros([Bd.shape[1], Bd.shape[0]]))
        self.epsilon = epsilon
    
    # 离散LQR迭代求解，给定一个P的初值，迭代更新P和K，直到收敛
    def solve(self, MAX_ITER = 10000):
        for it in range(MAX_ITER):
            self.K = (self.Bd.T * self.P * self.Bd).I * self.Bd.T * self.P * self.Ad
            A_tilde = self.Ad - self.Bd * self.K
            P_ = A_tilde.T * self.P * A_tilde + self.K.T * self.R * self.K + self.Q
            if(abs(P_ - self.P).max() < self.epsilon):
                break
            self.P = P_
        if (it == MAX_ITER - 1):
            print("Maximum iteration reached.")
            return False
        return True