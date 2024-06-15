import math
import matplotlib.pyplot as plt
import numpy as np
# import imageio
from vehicle_model import LateralKinematicModel
from vehicle_model import LateralDynamicModel
from lqr import DiscreteLQR

def vehicle_model_main():
    #vehicle = LateralKinematicModel(0.0, 0.0, 0.0, 2.0)
    vehicle = LateralDynamicModel(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    trajectory_x = []
    trajectory_y = []
    fig = plt.figure()

    # 保存动图用
    # i = 0
    # image_list = []  # 存储图片
    plt.figure(1)
    for i in range(500):
        plt.cla()
        plt.gca().set_aspect('equal', adjustable='box')
        vehicle.update(np.pi / 10)
        vehicle.draw_vehicle(plt)
        trajectory_x.append(vehicle.x_cartesian)
        trajectory_y.append(vehicle.y_cartesian)
        plt.plot(trajectory_x, trajectory_y, 'blue')
        plt.xlim(-15, 12)
        plt.ylim(-2.5, 21)
        plt.pause(0.001)
    #     i += 1
    #     if (i % 5) == 0:
    #         plt.savefig("temp.png")
    #         image_list.append(imageio.imread("temp.png"))
    #
    # imageio.mimsave("display.gif", image_list, duration=0.1)

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

def vehicle_dlqr_main():
    vehicle = LateralDynamicModel(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    A = vehicle.A()
    B = vehicle.B()

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

    print(np.linalg.eig(A - B * dlqr.K))

    # 初始状态
    np.random.seed(0)  # 为了可重复性
    x0 = np.mat(np.random.randn(2)).T

    # LQR控制下的系统状态更新
    x_ctrl = np.mat(np.zeros((2, num_steps + 1)))
    x_ctrl[:, 0] = x0
    for k in range(num_steps):
        u = -clqr.K * x_ctrl[:, k]
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

if __name__ == "__main__":
    vehicle_dlqr_main()
