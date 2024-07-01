import os
import sys
import math
import platform
import numpy as np
from celluloid import Camera # 保存动图时用，pip install celluloid
from matplotlib import pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
vehicle_model_dir = os.path.join(project_root, 'vehicle_model')
reference_line_dir = os.path.join(project_root, 'reference_line')
sys.path.append(vehicle_model_dir)
sys.path.append(reference_line_dir)

from vehicle_model import KinematicModel_Rear
from reference_line import MyReferencePath

# 是否保存动图
IS_SAVE_GIF = True

def stanley_main():
    # 时间范围
    T = 100
    dt = 0.1
    num_steps = int(T / dt)
    time = np.linspace(0, T, num_steps + 1).T

    reference_path = MyReferencePath()
    goal = reference_path.refer_path[-1, 0:2]
    trajectory_x = []
    trajectory_y = []
    trajectory_x_front = []
    trajectory_y_front = []
    fig = plt.figure(1)

    vehicle = KinematicModel_Rear(0.0, -3.0, 0.0, 2.0, dt)

    if True == IS_SAVE_GIF:
        camera = Camera(fig)

    for i in range(num_steps):
        # stanley路径跟踪算法
        # 1. 根据前轴中心找到参考路径上的匹配点
        # 2. 计算横向位置误差ey
        # 3. 计算横向位置误差引起的控制量delta_e
        # 4. 计算航向误差引起的控制量delta_psi
        # 5. 计算前轮转角delta_f = delta_e + delta_psi
        front_x_cartesian = vehicle.x_cartesian + vehicle.L * np.cos(vehicle.psi)
        front_y_cartesian = vehicle.y_cartesian + vehicle.L * np.sin(vehicle.psi)

        # 参考线轨迹部分
        e, k, psi_ref, s0 = reference_path.calc_track_error(front_x_cartesian, front_y_cartesian)

        ey = np.square((reference_path.refer_path[s0, 0] - front_x_cartesian) ** 2 + (reference_path.refer_path[s0, 1] - front_y_cartesian) ** 2)
        ld = vehicle.v / 1.0
        delta_e = np.arctan2(ey, ld)

        delta_psi = psi_ref - vehicle.psi

        delta_f = delta_e + delta_psi
        u = np.matrix([[0], [delta_f]])

        # stanley控制下的系统状态更新
        vehicle.model_update(u)

        # 显示动图
        trajectory_x.append(vehicle.x_cartesian)
        trajectory_y.append(vehicle.y_cartesian)
        trajectory_x_front.append(vehicle.x_cartesian + vehicle.L * np.cos(vehicle.psi))
        trajectory_y_front.append(vehicle.y_cartesian + vehicle.L * np.sin(vehicle.psi))

        if False == IS_SAVE_GIF:
            plt.cla()
        plt.gca().set_aspect('equal', adjustable='box')
        vehicle.draw_vehicle(plt)
        plt.plot(reference_path.refer_path[:, 0], reference_path.refer_path[:, 1], "-.b",  linewidth=1.0, label="course") # 参考线轨迹
        plt.plot(trajectory_x, trajectory_y, 'red') # 车辆轨迹
        plt.plot(trajectory_x_front, trajectory_y_front, 'green') # 车辆后轴中心轨迹
        plt.xlim(-5, 55)
        plt.ylim(-10, 20)
        if False == IS_SAVE_GIF:
            plt.pause(0.001)

        if True == IS_SAVE_GIF:
            camera.snap()

        # 判断是否到达最后一个点
        if np.linalg.norm([vehicle.x_cartesian, vehicle.y_cartesian] - goal) <= 1.0:
            print("reach goal")
            break
    if True == IS_SAVE_GIF:
        animation = camera.animate()
        if platform.system() == "Windows":
            animation.save(current_dir + '\\trajectory.gif', writer = 'imagemagick')
        elif platform.system() == "Linux":
            animation.save(current_dir + '/trajectory.gif', writer = 'imagemagick')

if __name__ == "__main__":

    # 以车辆后轴中心为中心的车辆运动学模型的纯跟踪控制算法控制演示demo
    stanley_main()