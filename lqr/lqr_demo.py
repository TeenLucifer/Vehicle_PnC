import sys
import os
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
vehicle_model_dir = os.path.join(project_root, 'vehicle_model')
reference_line_dir = os.path.join(project_root, 'reference_line')
sys.path.append(vehicle_model_dir)
sys.path.append(reference_line_dir)

from vehicle_model import KinematicModel_Rear
from reference_line import MyReferencePath
from lqr import DiscreteLQR

# 一个二阶不稳定系统的LQR控制演示demo
def dlqr_main():
    # 系统矩阵（这个demo的系统本身不稳定，需要引入反馈控制才能稳定）
    A = np.mat(np.array([[1, 2],
                  [0, -1]]))
    B = np.mat(np.array([[0], [1]]))

    I = np.mat(np.eye(2))

    # 时间范围
    T = 10
    dt = 0.01
    num_steps = int(T / dt)
    time = np.linspace(0, T, num_steps + 1).T

    nx = A.shape[0]
    nu = B.shape[1]
    Ad = np.eye(nx) + A * dt
    Bd = B * dt
    Q = np.mat(np.eye(nx))
    R = np.mat(np.eye(nu))
    dlqr = DiscreteLQR(Ad, Bd, Q, R)
    if(False == dlqr.solve()):
        print("LQR求解失败")
        return

    # 初始状态
    np.random.seed(0)  # 为了可重复性
    x0 = np.mat(np.random.randn(2)).T

    # LQR控制下的系统状态更新
    x_ctrl = np.mat(np.zeros((2, num_steps + 1)))
    x_ctrl[:, 0] = x0
    for k in range(num_steps):
        u = -dlqr.K * x_ctrl[:, k]
        x_ctrl[:, k + 1] = Ad * x_ctrl[:, k] + Bd * u

    # 无控制下的系统状态更新
    x_no_ctrl = np.mat(np.zeros((2, num_steps + 1)))
    x_no_ctrl[:, 0] = x0
    for k in range(num_steps):
        u = 0
        x_no_ctrl[:, k + 1] = Ad * x_no_ctrl[:, k] + Bd * u

    # 绘制结果
    # 带LQR控制的系统状态
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.asarray(x_ctrl[0, :]).reshape(-1), label='x1(t)')
    plt.plot(time, np.asarray(x_ctrl[1, :]).reshape(-1), label='x2(t)')
    plt.title('System State With Control Over Time')
    plt.xlabel('Time t')
    plt.ylabel('State x(t)')
    plt.legend()
    plt.grid(True)

    # 无控制的系统状态
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.asarray(x_no_ctrl[0, :]).reshape(-1), label='x1(t)')
    plt.plot(time, np.asarray(x_no_ctrl[1, :]).reshape(-1), label='x2(t)')
    plt.title('System State Without Control Over Time')
    plt.xlabel('Time t')
    plt.ylabel('State x(t)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 以车辆后轴中心为中心的车辆运动学模型的LQR控制演示demo
def vehicle_dlqr_main():
    # 时间范围
    T = 100
    dt = 0.1
    num_steps = int(T / dt)
    time = np.linspace(0, T, num_steps + 1).T

    reference_path = MyReferencePath()
    goal = reference_path.refer_path[-1, 0:2]
    trajectory_x = []
    trajectory_y = []
    fig = plt.figure(1)

    vehicle = KinematicModel_Rear(0.0, 0.0, 0.0, 2.0, dt)
    Q = np.mat(np.eye(vehicle.nx) * 3)
    R = np.mat(np.eye(vehicle.nu) * 2)
    dlqr = DiscreteLQR(np.mat(np.eye(vehicle.nx)), np.mat(np.eye(vehicle.nu)), Q, R)

    for i in range(num_steps):
        # 参考线轨迹部分
        e, k, psi_ref, s0 = reference_path.calc_track_error(vehicle.x_cartesian, vehicle.y_cartesian)
        delta_ref = math.atan2(vehicle.L * k, 1)

        # 数学模型更新，相当于建模
        A, B = vehicle.state_space_model(delta_ref, psi_ref)
        nx = A.shape[0]
        nu = B.shape[1]
        Ad = np.eye(nx) + A * dt
        Bd = B * dt

        # 更新LQR控制器的系统矩阵并求解最优控制量
        dlqr.update_mat(Ad, Bd, Q, R)
        if(False == dlqr.solve()):
            print("LQR求解失败")
            return
        u = -dlqr.K * (vehicle.state_X - reference_path.refer_path[s0, 0:3].reshape(-1, 1))
        u[1, 0] = u[1, 0] + delta_ref

        # LQR控制下的系统状态更新
        vehicle.model_update(u)

        # 显示动图
        trajectory_x.append(vehicle.x_cartesian)
        trajectory_y.append(vehicle.y_cartesian)

        plt.cla()
        plt.gca().set_aspect('equal', adjustable='box')
        vehicle.draw_vehicle(plt)
        plt.plot(reference_path.refer_path[:, 0], reference_path.refer_path[:, 1], "-.b",  linewidth=1.0, label="course") # 参考线轨迹
        plt.plot(trajectory_x, trajectory_y, 'red') # 车辆轨迹
        plt.xlim(-10, 110)
        plt.ylim(-10, 20)
        plt.pause(0.001)

        # 判断是否到达最后一个点
        if np.linalg.norm([vehicle.x_cartesian, vehicle.y_cartesian] - goal) <= 0.1:
            print("reach goal")
            break

if __name__ == "__main__":

    # 一个二阶不稳定系统的LQR控制演示demo
    # dlqr_main()

    # 以车辆后轴中心为中心的车辆运动学模型的LQR控制演示demo
    vehicle_dlqr_main()
