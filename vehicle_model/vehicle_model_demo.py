import math
import matplotlib.pyplot as plt
import numpy as np
# import imageio
from vehicle_model import LateralKinematicModel
from vehicle_model import LateralDynamicModel

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

if __name__ == "__main__":
    vehicle_model_main()

