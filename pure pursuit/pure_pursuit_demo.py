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
IS_SAVE_GIF = False

def pure_prusuit_main():
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

    vehicle = KinematicModel_Rear(0.0, -3.0, 0.0, 2.0, dt)

    if True == IS_SAVE_GIF:
        camera = Camera(fig)

    for i in range(num_steps):
        # 参考线轨迹部分
        e, k, psi_ref, s0 = reference_path.calc_track_error(vehicle.x_cartesian, vehicle.y_cartesian)

        # 纯跟踪控制算法
        # 1. 匹配预瞄点
        # 2. 计算朝向角alpha
        # 3. 计算后轮转弯半径R
        # 4. 计算前轮转向角\delta_f
        ld = vehicle.v * 1.0
        pre_aim_point = np.array([0, 0])
        for i in range(s0, len(reference_path.refer_path)):
            ref_pos = reference_path.refer_path[i, 0 : 2]
            veh_pos = np.array([vehicle.x_cartesian, vehicle.y_cartesian])
            if np.sum(np.square(ref_pos - veh_pos)) >= ld * ld:
                pre_aim_point = ref_pos
                break

        x_r = pre_aim_point[0]
        y_r = pre_aim_point[1]
        alpha = np.arctan2(y_r - vehicle.y_cartesian, x_r - vehicle.x_cartesian) - vehicle.psi

        R = ld / (2 * np.sin(alpha))

        delta_f = np.arctan2(vehicle.L, R)
        u = np.matrix([[0], [delta_f]])

        # MPC控制下的系统状态更新
        vehicle.model_update(u)

        # 显示动图
        trajectory_x.append(vehicle.x_cartesian)
        trajectory_y.append(vehicle.y_cartesian)

        if False == IS_SAVE_GIF:
            plt.cla()
        plt.gca().set_aspect('equal', adjustable='box')
        vehicle.draw_vehicle(plt)
        plt.plot(reference_path.refer_path[:, 0], reference_path.refer_path[:, 1], "-.b",  linewidth=1.0, label="course") # 参考线轨迹
        plt.plot(trajectory_x, trajectory_y, 'red') # 车辆后轴中心轨迹
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
    pure_prusuit_main()