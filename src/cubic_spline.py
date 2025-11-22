import math
import numpy as np
import bisect

class CubicSpline1D:
    def __init__(self, x, y):
        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted")

        self.a = [iy for iy in y]
        self.b = [0.0] * (len(x) - 1)
        self.c = [0.0] * (len(x) - 1)
        self.d = [0.0] * (len(x) - 1)
        self.x = x
        self.nx = len(x)  # dimension of x

        # Standard Tridiagonal Matrix Algorithm
        a_matrix = np.zeros((self.nx, self.nx))
        b_vector = np.zeros(self.nx)
        
        a_matrix[0, 0] = 1.0
        a_matrix[self.nx - 1, self.nx - 1] = 1.0

        for i in range(1, self.nx - 1):
            a_matrix[i, i - 1] = h[i - 1]
            a_matrix[i, i] = 2.0 * (h[i - 1] + h[i])
            a_matrix[i, i + 1] = h[i]
            b_vector[i] = 3.0 * (self.a[i + 1] - self.a[i]) / h[i] - 3.0 * (self.a[i] - self.a[i - 1]) / h[i - 1]

        # Solve c coefficients
        c = np.linalg.solve(a_matrix, b_vector)

        for i in range(self.nx - 1):
            self.c[i] = c[i]
            self.b[i] = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0
            self.d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

    def calc_position(self, x):
        if x < self.x[0]: return None
        if x > self.x[-1]: return None
        i = bisect.bisect(self.x, x) - 1
        i = min(max(i, 0), self.nx - 2)
        dx = x - self.x[i]
        return self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2 + self.d[i] * dx ** 3

    def calc_first_derivative(self, x):
        if x < self.x[0] or x > self.x[-1]: return None
        i = bisect.bisect(self.x, x) - 1
        i = min(max(i, 0), self.nx - 2)
        dx = x - self.x[i]
        return self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2

    def calc_second_derivative(self, x):
        if x < self.x[0] or x > self.x[-1]: return None
        i = bisect.bisect(self.x, x) - 1
        i = min(max(i, 0), self.nx - 2)
        dx = x - self.x[i]
        return 2.0 * self.c[i] + 6.0 * self.d[i] * dx

class CubicSpline2D:
    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(ds))
        return s

    def calc_position(self, s):
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)
        return x, y

    def calc_curvature(self, s):
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        # Avoid zero division
        den = (dx ** 2 + dy ** 2)**(1.5)
        if den == 0: return 0.0
        k = (ddy * dx - ddx * dy) / den
        return k

    def calc_yaw(self, s):
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        return math.atan2(dy, dx)