"""
Yaskawa GP8 Pick-and-Place Script for CoppeliaSim
-------------------------------------------------
This script calculates the joint angles required to move a robotic arm
through a sequence of Cartesian (XYZ) waypoints using Inverse Kinematics (IK),
generates smooth trajectories, and sends them to CoppeliaSim.
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
import numpy as np
import math
import time

# =====================================================================
# 1. USER CONFIGURATION & SETUP
# =====================================================================
# Move all settings up here so they are easy to find and change.

Pi = math.pi

# Timing and Simulation Settings
DT = 0.05               # Must match CoppeliaSim's simulation timestep
IK_STEPS = 40           # Number of waypoints to generate for smooth paths
G_CLOSE = -0.04       # Gripper state: +0.04 = open, -0.04 = close, 0.0 = hold
G_OPEN = 0.04        # Gripper state: +0.04 = open, -0.04 = close,
G_HOLD = 0.0         # Gripper state: +0.04 = open, -0.04 = close, 0.0 = hold
PLACE_DUR = 3.0         # How long each placing motion should take (seconds)

# Yaskawa GP8 Classical DH Parameters [a, alpha, d, theta_offset]
# These numbers define the physical lengths and twists of the robot's links.
DH_PARAMS = np.array([
    [0.040,  -Pi/2,   0.330,   0.0  ],
    [0.345,   0.0,    0.0,    -Pi/2 ],
    [0.040,  -Pi/2,   0.0,     0.0  ],
    [0.0,     Pi/2,   0.340,   0.0  ],
    [0.0,    -Pi/2,   0.0,     0.0  ],
    [0.0,     0.0,    0.24133, 0.0  ],
], dtype=float)

# Tool Center Point (TCP) Transform
# This shifts the calculation from the robot's wrist exactly to the tip of the gripper.
T_TOOL = np.array([
    [1, 0, 0,  0.00007],
    [0, 1, 0, -0.00023],
    [0, 0, 1,  0.01867],
    [0, 0, 0,  1.0    ],
], dtype=float)

# Joint Sign Map: +1 = DH model matches simulation, -1 = rotation is flipped in sim
JOINT_SIGN = np.array([+1, +1, -1, +1, -1, +1], dtype=float)

# ---------------- Waypoints ----------------

# The precise path the robot will follow to grab the object (Position only: X, Y, Z)
TARGETS = [
    np.array([-0.032915, -0.887067, 0.486025]),
    np.array([-0.03231, -0.882233, 0.48512]),
    np.array([-0.031694, -0.877359, 0.484205]),
    np.array([-0.031058, -0.872478, 0.483287]),
    np.array([-0.030413, -0.867584, 0.48237]),
    np.array([-0.029721, -0.862713, 0.481447]),
    np.array([-0.028934, -0.857858, 0.480519]),
    np.array([-0.028094, -0.852999, 0.47959]),
    np.array([-0.027244, -0.848145, 0.478659]),
    np.array([-0.026384, -0.843305, 0.477726]),
    np.array([-0.025498, -0.838449, 0.476796]),
    np.array([-0.024621, -0.833601, 0.475863]),
    np.array([-0.023648, -0.828783, 0.474924]),
    np.array([-0.022562, -0.823977, 0.473982]),
    np.array([-0.021453, -0.819179, 0.473037]),
    np.array([-0.020337, -0.814383, 0.472092]),
    np.array([-0.019201, -0.809595, 0.471147]),
    np.array([-0.018061, -0.804801, 0.470201]),
    np.array([-0.016866, -0.800026, 0.469252]),
    np.array([-0.015561, -0.795287, 0.4683]),
    np.array([-0.014181, -0.790559, 0.467345]),
    np.array([-0.012782, -0.78584, 0.466388]),
    np.array([-0.011369, -0.781133, 0.465431]),
    np.array([-0.009935, -0.776417, 0.464475]),
    np.array([-0.008503, -0.771708, 0.463517]),
    np.array([-0.006971, -0.767039, 0.462555]),
    np.array([-0.005307, -0.762407, 0.461592]),
    np.array([-0.003601, -0.757792, 0.460627]),
    np.array([-0.001883, -0.753182, 0.459662]),
    np.array([-0.000145, -0.74858, 0.458696]),
    np.array([0.001601, -0.743976, 0.45773]),
    np.array([0.00339, -0.739393, 0.456762]),
    np.array([0.005307, -0.734868, 0.455793]),
    np.array([0.007324, -0.730382, 0.454823]),
    np.array([0.009373, -0.725911, 0.453853]),
    np.array([0.01143, -0.721446, 0.452882]),
    np.array([0.013506, -0.716983, 0.451912]),
    np.array([0.015588, -0.712521, 0.450941]),
    np.array([0.017769, -0.708118, 0.449968]),
    np.array([0.020095, -0.703784, 0.448999]),
    np.array([0.022485, -0.699487, 0.448029]),
    np.array([0.02489, -0.695197, 0.447059]),
    np.array([0.027313, -0.690917, 0.446089]),
    np.array([0.029748, -0.68664, 0.445118]),
    np.array([0.032217, -0.682381, 0.444148]),
    np.array([0.03482, -0.678212, 0.443181]),
    np.array([0.037557, -0.674124, 0.442218]),
    np.array([0.040327, -0.670062, 0.441256]),
    np.array([0.043105, -0.666005, 0.440292]),
    np.array([0.045906, -0.661954, 0.439331]),
    np.array([0.048711, -0.657911, 0.438367]),
    np.array([0.051608, -0.653932, 0.437408]),
    np.array([0.054644, -0.650062, 0.436456]),
    np.array([0.057756, -0.646252, 0.435509]),
    np.array([0.060887, -0.642457, 0.434562]),
    np.array([0.064028, -0.638676, 0.433614]),
    np.array([0.067197, -0.634895, 0.432668]),
    np.array([0.07038, -0.63113, 0.431722]),
    np.array([0.073671, -0.627479, 0.430783]),
    np.array([0.077104, -0.623945, 0.429857]),
    np.array([0.080566, -0.620448, 0.428932]),
    np.array([0.084036, -0.61695, 0.428015]),
    np.array([0.087525, -0.61347, 0.427143]),
    np.array([0.091028, -0.609992, 0.426359]),
    np.array([0.094594, -0.60659, 0.425705]),
    np.array([0.098285, -0.603325, 0.425226]),
    np.array([0.102058, -0.600156, 0.424948]),
    np.array([0.10585, -0.597005, 0.424894]),
    np.array([0.109646, -0.593869, 0.425085]),
    np.array([0.113475, -0.590743, 0.425548]),
    np.array([0.117305, -0.587619, 0.426302]),
    np.array([0.121218, -0.584619, 0.427374]),
    np.array([0.125259, -0.581778, 0.428789]),
    np.array([0.129322, -0.57898, 0.430541]),
    np.array([0.133397, -0.576187, 0.43264]),
    np.array([0.137485, -0.573406, 0.435091]),
    np.array([0.141587, -0.570641, 0.437898]),
    np.array([0.145725, -0.567935, 0.441072]),
    np.array([0.149943, -0.565361, 0.444624]),
    np.array([0.154232, -0.562897, 0.448549]),
    np.array([0.158533, -0.560453, 0.452834]),
    np.array([0.162834, -0.558022, 0.457469]),
    np.array([0.167167, -0.55561, 0.462454]),
    np.array([0.1715, -0.553205, 0.467775]),
    np.array([0.175881, -0.55091, 0.473434]),
    np.array([0.180343, -0.548753, 0.47943]),
    np.array([0.184831, -0.54666, 0.485735]),
    np.array([0.189323, -0.544573, 0.492328]),
    np.array([0.193825, -0.5425, 0.499195]),
    np.array([0.198346, -0.540453, 0.50632]),
    np.array([0.202882, -0.538442, 0.51369]),
    np.array([0.207461, -0.536552, 0.521294]),
    np.array([0.212092, -0.534765, 0.529117]),
    np.array([0.216733, -0.533013, 0.537123]),
    np.array([0.221371, -0.531269, 0.545286]),
    np.array([0.226038, -0.529546, 0.553594]),
    np.array([0.230705, -0.527842, 0.562016]),
    np.array([0.235392, -0.526217, 0.570541]),
    np.array([0.240124, -0.524708, 0.579158]),
    np.array([0.244877, -0.523259, 0.587829]),
    np.array([0.249633, -0.521824, 0.596525]),
    np.array([0.254389, -0.52041, 0.605221]),
    np.array([0.259178, -0.519019, 0.613902]),
    np.array([0.263962, -0.517655, 0.622537]),
    np.array([0.268762, -0.51639, 0.631115]),
    np.array([0.2736, -0.515214, 0.639619]),
    np.array([0.278439, -0.514071, 0.648011]),
    np.array([0.283278, -0.512938, 0.656264]),
    np.array([0.288132, -0.511828, 0.664365]),
    np.array([0.293001, -0.510738, 0.672289]),
    np.array([0.297865, -0.509707, 0.680018]),
    np.array([0.30275, -0.50877, 0.687545]),
    np.array([0.307654, -0.50788, 0.694844]),
    np.array([0.312556, -0.50701, 0.701888]),
    np.array([0.317454, -0.506159, 0.708659]),
    np.array([0.322381, -0.505328, 0.715152]),
    np.array([0.327303, -0.504519, 0.721342]),
    np.array([0.332222, -0.503785, 0.727228]),
    np.array([0.337163, -0.503127, 0.732804]),
    np.array([0.342107, -0.502494, 0.738047]),
    np.array([0.347048, -0.50187, 0.742945]),
    np.array([0.351993, -0.501271, 0.747496]),
    np.array([0.356962, -0.500691, 0.751696]),
    np.array([0.361916, -0.500154, 0.755539]),
    np.array([0.366872, -0.499686, 0.759034]),
    np.array([0.371843, -0.499258, 0.76218]),
    np.array([0.376813, -0.498846, 0.76497]),
    np.array([0.381777, -0.498454, 0.767409]),
    np.array([0.386759, -0.498081, 0.76951]),
    np.array([0.391746, -0.49772, 0.771278]),
    np.array([0.396718, -0.497418, 0.77273]),
    np.array([0.401698, -0.497179, 0.77389]),
    np.array([0.406682, -0.496957, 0.774762]),
    np.array([0.411663, -0.496742, 0.775365]),
    np.array([0.416647, -0.49655, 0.775725]),
    np.array([0.421649, -0.496375, 0.775869]),
    np.array([0.426637, -0.496232, 0.775823]),
    np.array([0.431619, -0.496148, 0.775627]),
    np.array([0.436608, -0.496102, 0.775315]),
    np.array([0.441597, -0.496067, 0.774919]),
    np.array([0.446582, -0.496049, 0.774479]),
    np.array([0.451577, -0.496049, 0.774036]),
    np.array([0.456582, -0.496059, 0.773597]),
    np.array([0.461568, -0.496118, 0.773161]),
    np.array([0.466553, -0.496234, 0.772737]),
    np.array([0.471543, -0.496365, 0.772319]),
    np.array([0.47653, -0.496503, 0.7719]),
    np.array([0.481517, -0.496662, 0.771484]),
    np.array([0.48652, -0.496839, 0.771074]),
    np.array([0.491513, -0.497041, 0.770666]),
    np.array([0.496492, -0.497299, 0.770266]),
    np.array([0.501472, -0.4976, 0.769877]),
    np.array([0.506453, -0.497912, 0.76949]),
    np.array([0.511431, -0.498238, 0.769104]),
    np.array([0.516418, -0.498582, 0.768723]),
    np.array([0.521414, -0.498939, 0.768345]),
    np.array([0.526386, -0.499344, 0.767972]),
    np.array([0.531353, -0.499815, 0.767613]),
    np.array([0.536321, -0.500314, 0.767261]),
    np.array([0.541286, -0.500821, 0.76691]),
    np.array([0.546247, -0.501349, 0.766561]),
]

PLACE_POSITIONS = [
    np.array([0.865, 0.0, 0.7100]),   
    np.array([0.015, 0.5, 0.7100]),   
]

# Time for Phase 1 (Approach) + Time for Phase 2 (Grabbing)
APPROACH_DUR = IK_STEPS * DT
GRAB_DUR = len(TARGETS) * DT

TARGET_DUR = APPROACH_DUR + GRAB_DUR


# =====================================================================
# 2. DATA LOGGING & MATH
# =====================================================================

# Global storage for graphs
DATA_LOG = {
    't': [],
    'cup_pos': [], 'cup_vel': [],
    'cup_ori': [], 'cup_omega': [],
    'ee_pos': [], 'ee_vel': [],
    'ee_ori': [], 'ee_omega': [],
    'joint_pos': [],    # All 6 joint angles (sim frame, radians)
    'gripper_cmd': [],  # Actual gripper velocity command over time
}

# Gripper command tracking globals
_current_gripper_cmd  = G_HOLD   # scalar fallback used for constant phases
_dynamic_gripper_list = None     # set to the dynamic list during the grabbing phase
_dynamic_gripper_t0   = None     # sim time when the grabbing dispatch was sent

def log_simulation_data(sim, cup_handle, ee_handle):
    """Samples the physical state of objects directly from CoppeliaSim."""
    global _current_gripper_cmd, _dynamic_gripper_list, _dynamic_gripper_t0
    t = sim.getSimulationTime()
    
    # Avoid duplicate polling in the same simulation step
    if len(DATA_LOG['t']) > 0 and t <= DATA_LOG['t'][-1]:
        return

    # Grab Cup Data
    if cup_handle != -1:
        cp = sim.getObjectPosition(cup_handle, sim.handle_world)
        cv, cw = sim.getObjectVelocity(cup_handle)
        co = sim.getObjectOrientation(cup_handle, sim.handle_world)
    else:
        cp, cv, cw, co = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]

    # Grab End-Effector Data
    if ee_handle != -1:
        ep = sim.getObjectPosition(ee_handle, sim.handle_world)
        ev, ew = sim.getObjectVelocity(ee_handle)
        eo = sim.getObjectOrientation(ee_handle, sim.handle_world)
    else:
        ep, ev, ew, eo = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]

    # Log all 6 joint positions (sim frame)
    if 'joint_handles' in globals() and joint_handles:
        jp = [sim.getJointPosition(h) for h in joint_handles]
    else:
        jp = [0.0] * 6

    # Resolve current gripper command: interpolate from dynamic list if in grabbing phase,
    # otherwise use the scalar set for the current phase.
    if _dynamic_gripper_list is not None and _dynamic_gripper_t0 is not None:
        elapsed = t - _dynamic_gripper_t0
        idx = int(elapsed / DT)
        idx = max(0, min(idx, len(_dynamic_gripper_list) - 1))
        gripper_val = _dynamic_gripper_list[idx]
    else:
        gripper_val = _current_gripper_cmd

    DATA_LOG['t'].append(t)
    DATA_LOG['cup_pos'].append(cp)
    DATA_LOG['cup_vel'].append(cv)
    DATA_LOG['cup_ori'].append(co)
    DATA_LOG['cup_omega'].append(cw)
    DATA_LOG['ee_pos'].append(ep)
    DATA_LOG['ee_vel'].append(ev)
    DATA_LOG['ee_ori'].append(eo)
    DATA_LOG['ee_omega'].append(ew)
    DATA_LOG['joint_pos'].append(jp)
    DATA_LOG['gripper_cmd'].append(gripper_val)

def dh_matrix(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1.0 ],
    ])

def fk(q):
    T = np.eye(4)
    for i, (a, alpha, d, theta_off) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(a, alpha, d, q[i] + theta_off)
    T = T @ T_TOOL
    return T[:3, 3].copy(), T[:3, :3].copy()

def jacobian_pos(q, fd_eps=1e-7):
    Jv = np.zeros((3, 6))
    for i in range(6):
        q_p, q_m = q.copy(), q.copy()
        q_p[i] += fd_eps  
        q_m[i] -= fd_eps  
        p_p, _ = fk(q_p)
        p_m, _ = fk(q_m)
        Jv[:, i] = (p_p - p_m) / (2 * fd_eps)
    return Jv

def inverse_kinematics_pos(target_pos, initial_guess, max_iter=300, tol=1e-4, alpha=0.5, lambda_=0.01):
    q = np.array(initial_guess, dtype=float)
    for _ in range(max_iter):
        pos_curr, _ = fk(q)
        error = target_pos - pos_curr          
        if np.linalg.norm(error) < tol:        
            break
        Jv    = jacobian_pos(q)                
        A     = Jv @ Jv.T + lambda_**2 * np.eye(3) 
        J_dls = Jv.T @ np.linalg.inv(A)       
        q     = q + alpha * (J_dls @ error)
    return q

def dh_to_sim(q):
    return JOINT_SIGN * q


# =====================================================================
# 3. TRAJECTORY GENERATION (PATH PLANNING)
# =====================================================================

def resample_to_n(configs, n):
    src = len(configs)
    if src == n:
        return configs
    out = []
    for i in range(n):
        f  = i * (src - 1) / (n - 1)
        lo = int(f)
        hi = min(lo + 1, src - 1)
        t  = f - lo
        out.append((1.0 - t) * configs[lo] + t * configs[hi])
    return out

def build_cartesian_trajectory(pos_start, pos_target, q_seed, n_ik=IK_STEPS):
    configs = []
    q = q_seed.copy()
    for k in range(n_ik):
        t   = k / (n_ik - 1)
        s   = 6*t**5 - 15*t**4 + 10*t**3           
        pos = (1.0 - s) * pos_start + s * pos_target
        q   = inverse_kinematics_pos(pos, q)
        configs.append(q.copy())
    return configs

def build_joint_trajectory(q_start, q_target, n_steps=IK_STEPS):
    configs = []
    for k in range(n_steps):
        t = k / (n_steps - 1)
        s = 6*t**5 - 15*t**4 + 10*t**3           
        q = (1.0 - s) * q_start + s * q_target
        configs.append(q)
    return configs


# =====================================================================
# 4. COPPELIA SIMULATION HELPERS
# =====================================================================

TARGET_ARM   = '/yaskawa'
STR_SIGNAL   = TARGET_ARM + '_executedMovId'
_exec_mov_id = 'notReady'

def wait_for_movement(sim, move_id, cup_handle, ee_handle, timeout=60.0):
    """Pauses Python until the movement is done, polling logs in the background."""
    global _exec_mov_id
    _exec_mov_id = 'notReady'
    t0 = time.time()
    while _exec_mov_id != move_id:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Movement '{move_id}' timed out after {timeout} s")
        s = sim.getStringSignal(STR_SIGNAL)
        if s is not None:
            _exec_mov_id = s.decode() if isinstance(s, bytes) else s
            
        # Perform continuous logging while we wait!
        log_simulation_data(sim, cup_handle, ee_handle)
        
        time.sleep(0.05)

def dispatch(sim, configs_sim, times, move_id, gripper_vel=0.0):
    """
    MODIFIED: Now accepts a list for gripper_vel, allowing the gripper 
    state to change smoothly mid-movement.
    """
    assert len(configs_sim) == len(times), "configs_sim length != times length"
    
    # If gripper_vel is a single number, stretch it to cover all steps.
    # If it is already a list, use it directly.
    if isinstance(gripper_vel, list):
        g_list = [float(g) for g in gripper_vel]
    else:
        g_list = [float(gripper_vel)] * len(times)
    
    md = {
        'id':      move_id,
        'type':    'pts',
        'times':   times,
        'j1':      [float(q[0]) for q in configs_sim],
        'j2':      [float(q[1]) for q in configs_sim],
        'j3':      [float(q[2]) for q in configs_sim],
        'j4':      [float(q[3]) for q in configs_sim],
        'j5':      [float(q[4]) for q in configs_sim],
        'j6':      [float(q[5]) for q in configs_sim],
        'gripper': g_list,
    }
    sim.callScriptFunction(f'remoteApi_movementDataFunction@{TARGET_ARM}', sim.scripttype_childscript, md)
    sim.callScriptFunction(f'remoteApi_executeMovement@{TARGET_ARM}', sim.scripttype_childscript, move_id)


# =====================================================================
# 6. PLOTTING FUNCTION
# =====================================================================
def plot_results():
    print("\nGenerating Plots...")
    t = np.array(DATA_LOG['t'])
    if len(t) == 0:
        print("No data logged to plot.")
        return

    cup_pos = np.array(DATA_LOG['cup_pos'])
    cup_vel = np.array(DATA_LOG['cup_vel'])
    cup_ori = np.array(DATA_LOG['cup_ori'])
    cup_omega = np.array(DATA_LOG['cup_omega'])
    
    ee_pos = np.array(DATA_LOG['ee_pos'])
    ee_vel = np.array(DATA_LOG['ee_vel'])
    ee_ori = np.array(DATA_LOG['ee_ori'])
    ee_omega = np.array(DATA_LOG['ee_omega'])

    # -- 1. Cup Position and Velocity --
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig1.suptitle("1. Cup Trajectory over Time")
    ax1.plot(t, cup_pos[:,0], label='X')
    ax1.plot(t, cup_pos[:,1], label='Y')
    ax1.plot(t, cup_pos[:,2], label='Z')
    ax1.set_ylabel('Position (m)')
    ax1.legend(); ax1.grid(True)

    ax2.plot(t, cup_vel[:,0], label='Vx')
    ax2.plot(t, cup_vel[:,1], label='Vy')
    ax2.plot(t, cup_vel[:,2], label='Vz')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_xlabel('Time (s)')
    ax2.legend(); ax2.grid(True)

    # -- 2. End-Effector Linear Trajectory --
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
    fig2.suptitle("2. End-Effector Linear Trajectory")
    ax3.plot(t, ee_pos[:,0], label='X')
    ax3.plot(t, ee_pos[:,1], label='Y')
    ax3.plot(t, ee_pos[:,2], label='Z')
    ax3.set_ylabel('Position (m)')
    ax3.legend(); ax3.grid(True)

    ax4.plot(t, ee_vel[:,0], label='Vx')
    ax4.plot(t, ee_vel[:,1], label='Vy')
    ax4.plot(t, ee_vel[:,2], label='Vz')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_xlabel('Time (s)')
    ax4.legend(); ax4.grid(True)

    # -- 2b. End-Effector 3D Path --
    fig3 = plt.figure(figsize=(8, 8))
    ax5 = fig3.add_subplot(111, projection='3d')
    fig3.suptitle("2b. End-Effector 3D Path")
    ax5.plot(ee_pos[:,0], ee_pos[:,1], ee_pos[:,2], label='TCP Path', color='blue', linewidth=2)
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_zlabel('Z (m)')
    ax5.legend()

    # -- 3. End-Effector Angular Trajectory --
    fig4, (ax6, ax7) = plt.subplots(2, 1, figsize=(10, 8))
    fig4.suptitle("3. End-Effector Angular Trajectory (Orientation & Omega)")
    
    # Unwrap angles to prevent jumpy charts from -180 to +180 flips
    ee_ori_unwrapped = np.unwrap(ee_ori, axis=0) 

    ax6.plot(t, np.degrees(ee_ori_unwrapped[:,0]), label='Alpha (Roll)')
    ax6.plot(t, np.degrees(ee_ori_unwrapped[:,1]), label='Beta (Pitch)')
    ax6.plot(t, np.degrees(ee_ori_unwrapped[:,2]), label='Gamma (Yaw)')
    ax6.set_ylabel('Orientation (Degrees)')
    ax6.legend(); ax6.grid(True)

    ax7.plot(t, ee_omega[:,0], label='Omega X')
    ax7.plot(t, ee_omega[:,1], label='Omega Y')
    ax7.plot(t, ee_omega[:,2], label='Omega Z')
    ax7.set_ylabel('Angular Vel (rad/s)')
    ax7.set_xlabel('Time (s)')
    ax7.legend(); ax7.grid(True)
    
    # -- 4. Cup Angular Trajectory --
    fig5, (ax8, ax9) = plt.subplots(2, 1, figsize=(10, 8))
    fig5.suptitle("4. Cup Angular Trajectory (Orientation & Omega)")
    
    cup_ori_unwrapped = np.unwrap(cup_ori, axis=0) 

    ax8.plot(t, np.degrees(cup_ori_unwrapped[:,0]), label='Alpha (Roll)')
    ax8.plot(t, np.degrees(cup_ori_unwrapped[:,1]), label='Beta (Pitch)')
    ax8.plot(t, np.degrees(cup_ori_unwrapped[:,2]), label='Gamma (Yaw)')
    ax8.set_ylabel('Orientation (Degrees)')
    ax8.legend(); ax8.grid(True)

    ax9.plot(t, cup_omega[:,0], label='Omega X')
    ax9.plot(t, cup_omega[:,1], label='Omega Y')
    ax9.plot(t, cup_omega[:,2], label='Omega Z')
    ax9.set_ylabel('Angular Vel (rad/s)')
    ax9.set_xlabel('Time (s)')
    ax9.legend(); ax9.grid(True)

    # -- 5. All 6 Joint Profiles --
    joint_pos = np.array(DATA_LOG['joint_pos'])
    if joint_pos.ndim == 2 and joint_pos.shape[1] == 6:
        fig6, axes_j = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
        fig6.suptitle("5. Joint Angle Profiles vs Time", fontsize=13)
        colors_j = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        for idx in range(6):
            axes_j[idx].plot(t, np.degrees(joint_pos[:, idx]),
                             color=colors_j[idx], linewidth=1.5,
                             label=f'Joint {idx+1}')
            axes_j[idx].set_ylabel('Angle (°)', fontsize=8)
            axes_j[idx].legend(loc='upper right', fontsize=8)
            axes_j[idx].grid(True)
        axes_j[-1].set_xlabel('Time (s)')
        fig6.tight_layout(rect=[0, 0, 1, 0.97])

    # -- 6. Gripper Command vs Time --
    gripper_cmd = np.array(DATA_LOG['gripper_cmd'])
    fig7, ax_g = plt.subplots(figsize=(10, 4))
    fig7.suptitle("6. Gripper Command vs Time", fontsize=13)

    ax_g.plot(t, gripper_cmd, color='tab:blue', linewidth=2)
    ax_g.set_xlabel('Time (s)')
    ax_g.set_ylabel('Gripper Velocity Command')

    # Draw reference lines so the named states are easy to read
    ax_g.axhline(G_CLOSE, color='tab:red',    linewidth=1, linestyle=':', alpha=0.6, label=f'G_CLOSE ({G_CLOSE:+.3f})')
    ax_g.axhline(G_HOLD,  color='tab:orange', linewidth=1, linestyle=':', alpha=0.6, label=f'G_HOLD  ({G_HOLD:+.3f})')
    ax_g.axhline(G_OPEN,  color='tab:green',  linewidth=1, linestyle=':', alpha=0.6, label=f'G_OPEN  ({G_OPEN:+.3f})')

    ax_g.set_ylim(G_CLOSE - 0.01, G_OPEN + 0.01)
    ax_g.legend(fontsize=10)
    ax_g.grid(True, axis='y', linestyle=':')
    fig7.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

# =====================================================================
# 5. MAIN EXECUTION SCRIPT
# =====================================================================

if __name__ == "__main__":
    print("--- Connecting to CoppeliaSim ---")
    client = RemoteAPIClient()
    sim    = client.getObject('sim')
    joint_handles = [sim.getObject(f'{TARGET_ARM}/joint{i+1}') for i in range(6)]

    # Fetch Handles for Logging
    try:
        cup_handle = sim.getObject('./Cup')
    except:
        cup_handle = -1
        print("Warning: Object './Cup' not found. Cup data will be zeroed.")
        
    try:
        # Defaulting to gripperEF; if named differently, change it here
        ee_handle = sim.getObject('./gripperEF')
    except:
        ee_handle = -1
        print("Warning: Object './gripperEF' not found. EE data will be zeroed.")


    N_STEPS = int(TARGET_DUR / DT)
    times   = [i * DT for i in range(N_STEPS)]

    q_sim_current = np.array([sim.getJointPosition(h) for h in joint_handles])
    q_dh_current  = JOINT_SIGN * q_sim_current
    pos_start, _ = fk(q_dh_current)

    # -----------------------------------------------------------------
    # Phase 1: Smooth approach to the start of the grab
    # -----------------------------------------------------------------
    ik_configs_dh = [q_dh_current.copy()]
    q_seed = q_dh_current.copy()

    print("\n--- Phase 1: Calculating smooth Cartesian approach to the first waypoint ---")
    approach_configs = build_cartesian_trajectory(pos_start, TARGETS[0], q_seed, n_ik=IK_STEPS)
    ik_configs_dh.extend(approach_configs)
    q_seed = approach_configs[-1].copy()

    # -----------------------------------------------------------------
    # Phase 2: Solve Inverse Kinematics for the dense grabbing path
    # -----------------------------------------------------------------
    print(f"\n--- Phase 2: Solving IK for the remaining {len(TARGETS)-1} waypoints ---")
    for idx in range(1, len(TARGETS)):
        target_pos = TARGETS[idx]
        q_ik = inverse_kinematics_pos(target_pos, q_seed)
        ik_configs_dh.append(q_ik.copy())
        
        q_seed = q_ik.copy() 
        if idx % 10 == 0 or idx == len(TARGETS) - 1:
            print(f"  Solved {idx}/{len(TARGETS)-1} remaining waypoints...")

    ik_configs_sim = [dh_to_sim(q) for q in ik_configs_dh]
    configs_sim = resample_to_n(ik_configs_sim, N_STEPS)

    # -----------------------------------------------------------------
    # GRIPPER LOGIC: Calculate when to close the gripper
    # -----------------------------------------------------------------
    # We want it open/hold (0.0) until we finish the approach (reaching TARGETS[0]),
    # and closed (-0.04) while sweeping through the rest of the TARGETS array.
    
    # In ik_configs_dh, TARGETS[0] happens exactly at index `IK_STEPS`.
    fraction_to_target_0 = IK_STEPS / (len(ik_configs_dh) - 1)
    split_index = int(fraction_to_target_0 * N_STEPS)

    # Create dynamic gripper list: [0.0, ..., 0.0, -0.04, ..., -0.04]
    dynamic_gripper = [G_HOLD] * split_index + [G_CLOSE] * (N_STEPS - split_index)

    # -----------------------------------------------------------------
    # Phase 3: Calculate Joint Space Trajectories for all Placing points
    # -----------------------------------------------------------------
    print("\n--- Phase 3: Calculating Joint Space Trajectories for Placing ---")
    q_current_placing = ik_configs_dh[-1]  
    all_placing_trajectories = []

    for i, target_pos in enumerate(PLACE_POSITIONS):
        print(f"  Calculating path for Placing Position {i+1}/{len(PLACE_POSITIONS)}...")
        q_target_placing = inverse_kinematics_pos(target_pos, q_current_placing)
        
        placing_configs_dh = build_joint_trajectory(q_current_placing, q_target_placing, n_steps=IK_STEPS)
        placing_configs_sim = [dh_to_sim(q) for q in placing_configs_dh]
        
        N_STEPS_PLACE = int(PLACE_DUR / DT)
        times_place   = [k * DT for k in range(N_STEPS_PLACE)]
        configs_sim_place = resample_to_n(placing_configs_sim, N_STEPS_PLACE)
        
        all_placing_trajectories.append({
            'configs': configs_sim_place,
            'times': times_place,
            'id': f'waypoint_path_placing_{i+1}'
        })
        q_current_placing = q_target_placing

    # -----------------------------------------------------------------
    # Phase 4: Execute Simulation
    # -----------------------------------------------------------------
    print("\n--- Starting Simulation ---")
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_advancing_running:
        time.sleep(0.1)
    print("Simulation running.") 
    
    wait_for_movement(sim, 'ready', cup_handle, ee_handle)

    # Execute Grabbing (Gripper will close automatically after TARGETS[0])
    print(f"\nSending Grabbing trajectory to CoppeliaSim (Duration: {TARGET_DUR}s)...")
    _dynamic_gripper_list = dynamic_gripper
    _dynamic_gripper_t0   = sim.getSimulationTime()
    dispatch(sim, configs_sim, times, 'waypoint_path_grabbing', gripper_vel=dynamic_gripper)
    print("  Moving to Grabbing State...", end='', flush=True)
    wait_for_movement(sim, 'waypoint_path_grabbing', cup_handle, ee_handle)
    print("\r  Done.                         ")
    # Grabbing done — switch back to scalar tracking
    _dynamic_gripper_list = None
    _dynamic_gripper_t0   = None
    _current_gripper_cmd  = G_CLOSE

    # Execute Placing (Keep gripper closed = G_CLOSE = -0.04)
    print(f"\n--- Executing {len(all_placing_trajectories)} Placing Trajectories ---")
    for i, traj in enumerate(all_placing_trajectories):
        move_id = traj['id']
        print(f"Sending Placing trajectory {i+1} to CoppeliaSim (Duration: {PLACE_DUR}s)...")
        
        dispatch(sim, traj['configs'], traj['times'], move_id, gripper_vel=G_CLOSE)

        print(f"  Moving to Placing Target {i+1}...", end='', flush=True)
        wait_for_movement(sim, move_id, cup_handle, ee_handle)
        print("\r  Done.                         ")
        time.sleep(0.5) 
        
    # -----------------------------------------------------------------
    # Phase 5: Execute Rotation Placing
    # -----------------------------------------------------------------
    print("\n--- Executing Rotation Placing ---")
    
    linearConveyor = sim.getObject('/conveyor')
    LCpos = sim.getObjectPosition(linearConveyor, sim.handle_world)
    alpha, beta, gamma = sim.getObjectOrientation(linearConveyor, sim.handle_world)
    print(f"  Current Conveyor Position: {LCpos}, Orientation (radians): ({alpha:.2f}, {beta:.2f}, {gamma:.2f})")
    
    
    TARGET_J5_DEG = alpha   
    TARGET_J6_DEG = -beta
    
    q_start_rot = q_current_placing.copy()
    q_target_rot = q_start_rot.copy()
    
    q_target_rot[4] = TARGET_J5_DEG  # Joint 5 (index 4)
    q_target_rot[5] = TARGET_J6_DEG  # Joint 6 (index 5)
    
    rot_configs_dh = build_joint_trajectory(q_start_rot, q_target_rot, n_steps=IK_STEPS)
    rot_configs_sim = [dh_to_sim(q) for q in rot_configs_dh]
    
    ROT_DUR = 2.0  
    N_STEPS_ROT = int(ROT_DUR / DT)
    times_rot = [k * DT for k in range(N_STEPS_ROT)]
    configs_sim_rot = resample_to_n(rot_configs_sim, N_STEPS_ROT)
    
    move_id_rot = 'waypoint_path_rot_placing'
    print(f"Sending Rotation trajectory to CoppeliaSim (Duration: {ROT_DUR}s)...")
    
    # Send the rotation command while KEEPING THE GRIPPER CLOSED
    dispatch(sim, configs_sim_rot, times_rot, move_id_rot, gripper_vel=G_CLOSE)
    
    print(f"  Rotating J5 to {TARGET_J5_DEG} rad and J6 to {TARGET_J6_DEG} rad...", end='', flush=True)
    wait_for_movement(sim, move_id_rot, cup_handle, ee_handle)
    print("\r  Done.                                                     ")
    time.sleep(0.5)
    
    # --- NEW: OPEN GRIPPER AFTER ROTATION FINISHES ---
    move_id_open = 'waypoint_path_open_gripper'
    OPEN_DUR = 1.0  
    N_STEPS_OPEN = int(OPEN_DUR / DT)
    times_open = [k * DT for k in range(N_STEPS_OPEN)]
    
    # Create a stationary trajectory (staying exactly where we ended the rotation)
    configs_sim_open = [configs_sim_rot[-1]] * N_STEPS_OPEN
    
    print(f"Sending Open Gripper command to CoppeliaSim (Duration: {OPEN_DUR}s)...")
    _current_gripper_cmd = G_OPEN
    dispatch(sim, configs_sim_open, times_open, move_id_open, gripper_vel=G_OPEN)
    
    print("  Opening Gripper...", end='', flush=True)
    wait_for_movement(sim, move_id_open, cup_handle, ee_handle)
    print("\r  Gripper Opened.                                           ")
    time.sleep(0.5)

    if (input("\nPress Enter to stop the simulation...") is not None):
        sim.stopSimulation()
        print("\nSimulation stopped.")
        
    # Generate the requested graphs
    plot_results()