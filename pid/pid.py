class PID:
    """位置式实现
    """
    def __init__(self, upper, lower, k = [1., 0., 0.]):
        self.kp, self.ki, self.kd = k

        self.e = 0.0      # error
        self.pre_e = 0.0  # previous error
        self.sum_e = 0.0  # sum of error

        self.upper_bound = upper    # upper bound of output
        self.lower_bound = lower    # lower bound of output

    def set_param(self, k, upper, lower):
        self.kp, self.ki, self.kd = k
        self.upper_bound = upper
        self.lower_bound = lower

    def cal_output(self, ref, feedback):   # calculate output
        self.e = ref - feedback

        pid_out = self.e * self.kp + self.sum_e * self.ki + (self.e - self.pre_e) * self.kd
        if pid_out < self.lower_bound:
            pid_out = self.lower_bound
        elif pid_out > self.upper_bound:
            pid_out = self.upper_bound

        self.pre_e = self.e
        self.sum_e += self.e

        return pid_out