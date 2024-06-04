import numpy as np
import math

class VehicleBaseModel:
    def __init__(self, L = 2.9, width = 2.0, LF = 3.8, LB = 0.8, TR = 0.5, TW = 0.5, Iz = 2250.0, Cf = 1600.0 * 1.0, Cr = 1700.0 * 1.0, m = 1500.0, MWA = 30.0):
         # 车辆参数信息
        self.L = L                      # 轴距[m]
        self.Lf = self.L / 2.0          # 车辆中心点到前轴的距离[m]
        self.Lr = self.L - self.Lf      # 车辆终点到后轴的距离[m]
        self.width = width              # 宽度[m]
        self.LF = LF                    # 后轴中心到车头距离[m]
        self.LB = LB                    # 后轴中心到车尾距离[m]
        self.TR = TR                    # 轮子半径[m]
        self.TW = TW                    # 轮子宽度[m]
        self.WD = self.width            # 轮距[m]
        self.Iz = Iz                    # 车辆绕z轴的转动惯量[kg/m2]
        self.Cf = Cf                    # 前轮侧偏刚度[N/rad]
        self.Cr = Cr                    # 后轮侧偏刚度[N/rad]
        self.m = m                      # 车身质量[kg]
        self.len = self.LB + self.LF    # 车辆长度[m]
        self.MWA = np.radians(MWA)      # 最大轮转角(Max Wheel Angle)[rad]

        self.x_cartesian = 0.0          # 车辆笛卡尔坐标系x坐标
        self.y_cartesian = 0.0          # 车辆笛卡尔坐标系y坐标
        self.psi = 0.0                  # 航向角
        self.delta = 0.0                # 前轮转向角
        self.dt = 0.1                   # 离散时间

    def normalize_angle(self, angle):
        a = math.fmod(angle + np.pi, 2 * np.pi)
        if a < 0.0:
            a += (2.0 * np.pi)
        return a - np.pi

    def draw_vehicle(self, axis, color='black'):
        vehicle_outline = np.array(
            [[-self.LB, self.LF, self.LF, -self.LB, -self.LB],
             [self.width / 2, self.width / 2, -self.width / 2, -self.width / 2, self.width / 2]])

        wheel = np.array([[-self.TR, self.TR, self.TR, -self.TR, -self.TR],
                          [self.TW / 2, self.TW / 2, -self.TW / 2, -self.TW / 2, self.TW / 2]])

        rr_wheel = wheel.copy()  # 右后轮
        rl_wheel = wheel.copy()  # 左后轮
        fr_wheel = wheel.copy()  # 右前轮
        fl_wheel = wheel.copy()  # 左前轮
        rr_wheel[1, :] += self.WD/2
        rl_wheel[1, :] -= self.WD/2

        # 方向盘旋转
        rot1 = np.array([[np.cos(self.delta), -np.sin(self.delta)],
                         [np.sin(self.delta), np.cos(self.delta)]])
        # psi旋转矩阵
        rot2 = np.array([[np.cos(self.psi), -np.sin(self.psi)],
                         [np.sin(self.psi), np.cos(self.psi)]])
        fr_wheel = np.dot(rot1, fr_wheel)
        fl_wheel = np.dot(rot1, fl_wheel)
        fr_wheel += np.array([[self.L], [-self.WD / 2]])
        fl_wheel += np.array([[self.L], [self.WD / 2]])

        fr_wheel = np.dot(rot2, fr_wheel)
        fr_wheel[0, :] += self.x_cartesian
        fr_wheel[1, :] += self.y_cartesian
        fl_wheel = np.dot(rot2, fl_wheel)
        fl_wheel[0, :] += self.x_cartesian
        fl_wheel[1, :] += self.y_cartesian
        rr_wheel = np.dot(rot2, rr_wheel)
        rr_wheel[0, :] += self.x_cartesian
        rr_wheel[1, :] += self.y_cartesian
        rl_wheel = np.dot(rot2, rl_wheel)
        rl_wheel[0, :] += self.x_cartesian
        rl_wheel[1, :] += self.y_cartesian
        vehicle_outline = np.dot(rot2, vehicle_outline)
        vehicle_outline[0, :] += self.x_cartesian
        vehicle_outline[1, :] += self.y_cartesian

        axis.plot(fr_wheel[0, :], fr_wheel[1, :], color)
        axis.plot(rr_wheel[0, :], rr_wheel[1, :], color)
        axis.plot(fl_wheel[0, :], fl_wheel[1, :], color)
        axis.plot(rl_wheel[0, :], rl_wheel[1, :], color)

        axis.plot(vehicle_outline[0, :], vehicle_outline[1, :], color)
        # ax.axis('equal')

# 车辆侧向动力学模型
# 输入：前轮转向角delta
class LateralKinematicModel(VehicleBaseModel):
    def __init__(self, x = 0.0, y = 0.0, psi = 0.0, v = 0.0):
        super().__init__()

        self.v = v
        self.dt = 0.1
    
    def update(self, delta):
        self.delta = np.clip(delta, -self.MWA, self.MWA)

        beta = math.atan(math.tan(delta) * self.Lf / (self.Lf + self.Lr))
        x_dot = self.v * math.cos(beta + self.psi)
        y_dot = self.v * math.sin(beta + self.psi)
        psi_dot = self.v * math.cos(beta) * math.tan(delta)

        self.x_cartesian = self.x_cartesian + x_dot * self.dt
        self.y_cartesian = self.y_cartesian + y_dot * self.dt
        self.psi = self.psi + psi_dot * self.dt
        self.psi = self.normalize_angle(self.psi)

# 车辆侧向动力学模型
# 输入：前轮转向角delta
class LateralDynamicModel(VehicleBaseModel):
    def __init__(self, x_car = 0.0, y_car = 0.0, vx = 0.01, vy = 0.0, ax = 0.0, ay = 0.0, dpsi = 0.0, ddpsi = 0.0):
        super().__init__()

        self.vx = vx              # 车辆坐标系下x方向速度
        self.vy = vy              # 车辆坐标系下y方向速度
        self.ax = ax              # 车辆坐标系下x方向加速度
        self.ay = ay              # 车辆坐标系下y方向加速度
        self.dpsi = dpsi          # 车辆航线角速度
        self.ddpsi = ddpsi        # 车辆航线角加速度

    def update(self, delta):
        self.delta = np.clip(delta, -self.MWA, self.MWA)

        #########################
        theta_vf = (self.vy + self.Lf * self.dpsi) / self.vx
        theta_vr = (self.vy - self.Lr * self.dpsi) / self.vx
        alpha_f = self.delta - theta_vf
        alpha_r = -theta_vr
        alpha_f = self.normalize_angle(alpha_f)
        alpha_r = self.normalize_angle(alpha_r)
        F_yf = 2 * self.Cf * alpha_f
        F_yr = 2 * self.Cr * alpha_r

        self.ay = (F_yf + F_yr) / self.m - self.vx * self.dpsi
        self.ddpsi = (self.Lf * F_yf - self.Lr * F_yr) / self.Iz

        self.vx = self.vx + self.ax * self.dt
        self.vy = self.vy + self.ay * self.dt
        self.dpsi = self.dpsi + self.ddpsi * self.dt

        self.x_cartesian = self.x_cartesian + self.vx * math.cos(self.psi) * self.dt - self.vy * math.sin(self.psi) * self.dt
        self.y_cartesian = self.y_cartesian + self.vx * math.sin(self.psi) * self.dt + self.vy * math.cos(self.psi) * self.dt
        self.psi = self.psi + self.dpsi * self.dt
        self.psi = self.normalize_angle(self.psi)

    def set_vx(self, vx):
        self.vx = vx

    def set_ax(self, ax):
        self.ax = ax