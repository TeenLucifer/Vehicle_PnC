import math
import numpy as np

class DiscreteLQR:
    def __init__(self, Ad, Bd, Q, R, epsilon = 1e-4):
        self.Ad = Ad # 离散系统矩阵
        self.Bd = Bd # 离散系统矩阵
        self.Q = Q
        self.R = R
        self.P = Q # 给个初始值Q
        self.K = np.matrix(np.zeros([Bd.shape[1], Bd.shape[0]]))
        self.epsilon = epsilon
    
    # 离散LQR迭代求解，给定一个P的初值，迭代更新P和K，直到收敛
    def solve(self, MAX_ITER = 100):
        self.P = self.Q
        for it in range(MAX_ITER):
            P_ = self.Q + self.Ad.T * self.P * self.Ad - self.Ad.T * self.P * self.Bd * (self.R + self.Bd.T * self.P * self.Bd).I * self.Bd.T * self.P * self.Ad
            if (abs(P_ - self.P).max() < self.epsilon):
                break
            self.P = P_
        if (it == MAX_ITER - 1):
            print("Maximum iteration reached.")
            return False
        self.K = (self.R + self.Bd.T * self.P * self.Bd).I * self.Bd.T * self.P * self.Ad
        return True

    def update_mat(self, Ad, Bd, Q, R):
        self.Ad = Ad
        self.Bd = Bd
        self.Q = Q
        self.R = R