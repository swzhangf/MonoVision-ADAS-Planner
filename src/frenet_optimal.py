import numpy as np
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from polynomials import QuinticPolynomial, QuarticPolynomial

MAX_SPEED = 80.0 / 3.6  # Maximum speed 80km/h
MAX_ACCEL = 3.0
MAX_CURVATURE = 1.0
MAX_ROAD_WIDTH = 10.0   # Road width widened to accommodate visual errors
D_ROAD_W = 1.0
DT = 0.2
MAX_T = 5.0
MIN_T = 4.0
TARGET_SPEED = 60.0 / 3.6
D_T_S = 5.0 / 3.6
N_S_SAMPLE = 1
ROBOT_RADIUS = 2.5      # Slightly increase vehicle radius as a safety margin

# Cost weights
KJ = 0.1
KT = 0.1
KD = 1.0
KLAT = 1.0
KLON = 1.0

class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0
        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []
    # Loop over target lateral deviation (d)
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):
        # Loop over planning time (T)
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()
            # Quintic polynomial for lateral motion
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]
            
            # Loop over target speed (v)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE, TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = FrenetPath()
                # Copy lateral path information
                tfp.t, tfp.d, tfp.d_d, tfp.d_dd, tfp.d_ddd = fp.t, fp.d, fp.d_d, fp.d_dd, fp.d_ddd
                
                # Quartic polynomial for longitudinal motion
                lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)
                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
                
                # Calculate costs
                Jp = sum(np.power(tfp.d_ddd, 2)) # Lateral Jerk cost
                Js = sum(np.power(tfp.s_ddd, 2)) # Longitudinal Jerk cost
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2 # Speed deviation cost
                
                tfp.cd = KJ * Jp + KT * Ti + KD * tfp.d[-1] ** 2 # Lateral cost
                tfp.cv = KJ * Js + KT * Ti + KD * ds # Longitudinal cost
                tfp.cf = KLAT * tfp.cd + KLON * tfp.cv # Total cost
                
                frenet_paths.append(tfp)
                
    return frenet_paths

def calc_global_paths(fplist, csp):
    for fp in fplist:
        # Convert Frenet coordinates (s, d) to Global coordinates (x, y)
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None: break
            iyaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            
            # Transformation using the normal vector
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)
            
        if len(fp.x) < 2: continue # Skip invalid paths
        
        # Calculate yaw and path segment length (ds)
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))
            
        # Append last element to match length
        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])
        
    return fplist

def check_collision(fp, ob):
    if ob is None or len(ob) == 0: return True
    for i in range(len(ob[:, 0])):
        # Simple distance check
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2) for (ix, iy) in zip(fp.x, fp.y)]
        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])
        if collision: return False
        
    return True

def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):
    # 1. Generate candidate Frenet paths
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    
    # 2. Convert Frenet paths to Global paths
    fplist = calc_global_paths(fplist, csp)
    
    # 3. Check for collisions (and other constraints, if any)
    fplist = [fp for fp in fplist if check_collision(fp, ob)]
    
    # 4. Find the path with the minimum cost
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp
            
    return best_path