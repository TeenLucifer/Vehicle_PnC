import numpy as np
import matplotlib.pyplot as plt

# 参数设置
end_time = 100
t_list = np.arange(0, end_time, 1)  # 时间向量

# 用一个匀速直线运动的模型
# 没有模型直接一个信号的话，无法进行卡尔曼滤波
# 系统状态x=[s, v]
# 离散时间delta_t=0.01
A = np.matrix([[1, 0.01],
              [0,    1]])  # 状态转移矩阵
H = np.matrix([[1, 0]])  # 观测矩阵
nx = 2

# 噪声协方差矩阵
Q = np.matrix([[1e-4, 0],
              [0, 1e-4]])  # 过程噪声协方差矩阵
R = np.matrix([[0.1]])     # 观测噪声协方差矩阵

x     = np.matrix([[0], [10]])     # 系统初始状态
x_hat = np.matrix([[0], [10]])     # 估计初始状态
P     = np.matrix(np.eye(nx)) # 初始协方差矩阵

x_list = []
y_list = []
y_noised_list = []
y_filtered_list = []

for i in range(end_time):
    # 生成过程噪声和观测噪声
    system_noise = np.matrix(np.random.multivariate_normal([0, 0], Q)).transpose()
    measure_noise = np.random.normal(0, np.sqrt(R))
    x = A * x + system_noise
    z = H * x + measure_noise

    x_hat_ = A * x_hat
    P_     = A * P * A.transpose() + Q
    K      = P_ * H.transpose() * np.linalg.inv(H * P_ * H.transpose() + R)
    x_hat  = x_hat_ + K * (z - H * x_hat_)
    P      = (np.eye(2) - K * H) * P_

    y_list.append(x[0][0, 0])
    y_noised_list.append(z[0][0, 0])
    y_filtered_list.append(x_hat[0][0, 0])

# 绘制信号
plt.figure()
plt.plot(t_list, y_noised_list, label='Measured Signel(with noise)')
plt.plot(t_list, y_list, label='Original Signal(true value)', linestyle='--')
plt.plot(t_list, y_filtered_list, label='Filtered Signal')
plt.xlabel('t [s]')
plt.ylabel('signal value')
plt.title('Kalman Filter demo.')
plt.legend()
plt.show()