"""
Yaskawa GP8 — IK Solver (Position Only) → CoppeliaSim
=======================================================
Solves IK for a target TCP position only (no orientation constraint),
then sends the resulting joint trajectory to CoppeliaSim.

Usage
-----
Edit TARGET_POS at the top, then run with CoppeliaSim open:
    python ik_to_sim_pos_only.py
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math
import time

Pi = math.pi

# =====================================================================
# 1.  DH TABLE  (Classical DH, Yaskawa GP8)
# =====================================================================
DH_PARAMS = np.array([
    [0.040,  -Pi/2,   0.330,   0.0  ],
    [0.345,   0.0,    0.0,    -Pi/2 ],
    [0.040,  -Pi/2,   0.0,     0.0  ],
    [0.0,     Pi/2,   0.340,   0.0  ],
    [0.0,    -Pi/2,   0.0,     0.0  ],
    [0.0,     0.0,    0.24133, 0.0  ],
], dtype=float)

T_TOOL = np.array([
    [1, 0, 0,  0.00007],
    [0, 1, 0, -0.00023],
    [0, 0, 1,  0.01867],
    [0, 0, 0,  1.0    ],
], dtype=float)

# =====================================================================
# 2.  FORWARD KINEMATICS
# =====================================================================
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
    """Forward kinematics. Returns (position [3], rotation [3x3])."""
    T = np.eye(4)
    for i, (a, alpha, d, theta_off) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(a, alpha, d, q[i] + theta_off)
    T = T @ T_TOOL
    return T[:3, 3].copy(), T[:3, :3].copy()

# =====================================================================
# 3.  JACOBIAN  (position rows only — top 3 of full 6x6)
# =====================================================================
FD_EPS = 1e-7

def jacobian_pos(q):
    """
    3×6 positional Jacobian only.
    Each column = finite-difference of TCP position w.r.t. joint i.
    """
    Jv = np.zeros((3, 6))
    for i in range(6):
        q_p, q_m = q.copy(), q.copy()
        q_p[i] += FD_EPS
        q_m[i] -= FD_EPS
        p_p, _ = fk(q_p)
        p_m, _ = fk(q_m)
        Jv[:, i] = (p_p - p_m) / (2 * FD_EPS)
    return Jv

# =====================================================================
# 4.  MATH HELPERS
# =====================================================================
def R_to_rpy(R):
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    if abs(np.cos(pitch)) < 1e-6:
        roll = 0.0
        yaw  = np.arctan2(-R[1,2], R[1,1])
    else:
        roll = np.arctan2(R[2,1], R[2,2])
        yaw  = np.arctan2(R[1,0], R[0,0])
    return roll, pitch, yaw

# =====================================================================
# 5.  IK SOLVER — POSITION ONLY  (Damped Least Squares, 3×6 Jacobian)
# =====================================================================
def inverse_kinematics_pos(target_pos, initial_guess,
                            max_iter=300, tol=1e-4, alpha=0.5, lambda_=0.01):
    """
    Position-only Damped Least Squares IK.
    Orientation is completely unconstrained — the solver only minimises
    the 3D position error.

    Parameters
    ----------
    target_pos    : array (3,)  desired TCP position in metres
    initial_guess : array (6,)  starting joint angles in radians
    max_iter      : int         maximum iterations
    tol           : float       convergence tolerance (position error norm)
    alpha         : float       step size
    lambda_       : float       damping factor

    Returns
    -------
    q : array (6,)  joint angles in radians
    """
    q = np.array(initial_guess, dtype=float)
    for _ in range(max_iter):
        pos_curr, _ = fk(q)
        error = target_pos - pos_curr          # 3-vector, position only

        if np.linalg.norm(error) < tol:
            break

        Jv    = jacobian_pos(q)                # 3×6
        A     = Jv @ Jv.T + lambda_**2 * np.eye(3)
        J_dls = Jv.T @ np.linalg.inv(A)       # 6×3
        q     = q + alpha * (J_dls @ error)
    return q

# =====================================================================
# 6.  SIGN CONVERSION
# =====================================================================
def dh_to_sim(q):
    return JOINT_SIGN * q

# =====================================================================
# 7.  TRAJECTORY HELPERS
# =====================================================================
def resample_to_n(configs, n):
    """Resample a list of joint configs to exactly n points (linear interp)."""
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

# IK waypoints along the straight-line Cartesian path
IK_STEPS = 40
def build_cartesian_trajectory(pos_start, pos_target, q_seed, n_ik=IK_STEPS):
    """
    Interpolate n_ik Cartesian waypoints from pos_start to pos_target,
    solve position-only IK at each, return list of DH joint configs.
    """
    configs = []
    q = q_seed.copy()
    for k in range(n_ik):
        t   = k / (n_ik - 1)
        s   = 3*t**2 - 2*t**3              # smoothstep timing
        pos = (1.0 - s) * pos_start + s * pos_target
        q   = inverse_kinematics_pos(pos, q)
        configs.append(q.copy())
        print(f"  IK waypoint {k+1:3d}/{n_ik}  pos={np.round(pos,4)}  "
              f"q_deg={np.round(np.degrees(q),2).tolist()}")
    return configs

# =====================================================================
# 8.  COPPELIA HELPERS
# =====================================================================
TARGET_ARM   = '/yaskawa'
STR_SIGNAL   = TARGET_ARM + '_executedMovId'
_exec_mov_id = 'notReady'

def wait_for_movement(sim, move_id, timeout=60.0):
    global _exec_mov_id
    _exec_mov_id = 'notReady'
    t0 = time.time()
    while _exec_mov_id != move_id:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Movement '{move_id}' timed out after {timeout} s")
        s = sim.getStringSignal(STR_SIGNAL)
        if s is not None:
            _exec_mov_id = s.decode() if isinstance(s, bytes) else s
        time.sleep(0.05)

def dispatch(sim, configs_sim, times, move_id, gripper_vel=0.0):
    """Pack and send N_STEPS sim-space configs to CoppeliaSim."""
    assert len(configs_sim) == len(times), \
        f"configs_sim length {len(configs_sim)} != times length {len(times)}"
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
        'gripper': [float(gripper_vel)] * len(times),
    }
    sim.callScriptFunction(
        f'remoteApi_movementDataFunction@{TARGET_ARM}',
        sim.scripttype_childscript, md)
    sim.callScriptFunction(
        f'remoteApi_executeMovement@{TARGET_ARM}',
        sim.scripttype_childscript, move_id)




# =====================================================================
# 9.  MAIN
# =====================================================================

# ── Connect ───────────────────────────────────────────────────────
client = RemoteAPIClient()
sim    = client.getObject('sim')

# =====================================================================
# USER CONFIG — edit these
# =====================================================================

TCP = sim.getObjectPosition(sim.getObject('/gripperEF'))

# Target TCP position (orientation is FREE — IK ignores it)
TARGET_POS    = np.array(TCP)   # metres (x, y, z)

TARGETS = [
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
np.array([0.084036, -0.61695, 0.428007]),
np.array([0.087525, -0.61347, 0.427083]),
np.array([0.091028, -0.609992, 0.42616]),
np.array([0.094594, -0.60659, 0.425242]),
np.array([0.098285, -0.603325, 0.424339]),
np.array([0.102058, -0.600156, 0.423445]),
np.array([0.10585, -0.597005, 0.422554]),
np.array([0.109646, -0.593869, 0.421661]),
np.array([0.113475, -0.590743, 0.420771]),
np.array([0.117305, -0.587619, 0.419881]),
np.array([0.121218, -0.584619, 0.419002]),
np.array([0.125259, -0.581778, 0.418144]),
np.array([0.129322, -0.57898, 0.417289]),
np.array([0.133397, -0.576187, 0.416436]),
np.array([0.137485, -0.573406, 0.415583]),
np.array([0.141587, -0.570641, 0.41473]),
np.array([0.145725, -0.567935, 0.413884]),
np.array([0.149943, -0.565361, 0.413055]),
np.array([0.154232, -0.562897, 0.41224]),
np.array([0.158533, -0.560453, 0.411428])
]


# Initial joint guess for IK seed (radians)
INITIAL_GUESS = np.zeros(6)

# Joint sign map: +1 = DH same as sim, -1 = flipped
JOINT_SIGN = np.array([+1, +1, -1, +1, -1, +1], dtype=float)

# Gripper: +0.04 = open, -0.04 = close, 0.0 = hold
GRIPPER_VEL = 0.0

# Motion duration (seconds) and timestep
TARGET_DUR = 5.0
DT         = 0.05      # must match CoppeliaSim simulation timestep

joint_handles = [sim.getObject(f'{TARGET_ARM}/joint{i+1}') for i in range(6)]

N_STEPS = int(TARGET_DUR / DT)
times   = [i * DT for i in range(N_STEPS)]

# ── Start simulation ──────────────────────────────────────────────
sim.startSimulation()
while sim.getSimulationState() != sim.simulation_advancing_running:
    time.sleep(0.1)
print("Simulation running.")
wait_for_movement(sim, 'ready')
# ── Read current state ────────────────────────────────────────────
q_sim_current = np.array([sim.getJointPosition(h) for h in joint_handles])
q_dh_current  = JOINT_SIGN * q_sim_current

# Get the actual starting Cartesian position of the robot
pos_start, _ = fk(q_dh_current)

# We will store the full joint-space path here
ik_configs_dh = [q_dh_current.copy()]
q_seed = q_dh_current.copy()

print("\n--- Phase 1: Smooth approach to the first waypoint ---")
# Use your existing function to interpolate from current pos to TARGETS[0]
approach_configs = build_cartesian_trajectory(
    pos_start  = pos_start,
    pos_target = TARGETS[0],
    q_seed     = q_seed,
    n_ik       = IK_STEPS  # Default is 40 steps from your helper functions
)

# Add the approach path to our main trajectory list
ik_configs_dh.extend(approach_configs)

# Update the IK seed to the last position of the approach path
q_seed = approach_configs[-1].copy()

print(f"\n--- Phase 2: Solving IK for the remaining {len(TARGETS)-1} waypoints ---")
# Loop through the REST of the targets, starting from index 1
for idx in range(1, len(TARGETS)):
    target_pos = TARGETS[idx]
    q_ik = inverse_kinematics_pos(target_pos, q_seed)
    ik_configs_dh.append(q_ik.copy())
    q_seed = q_ik.copy() # Update seed for the next point
    
    if idx % 10 == 0 or idx == len(TARGETS) - 1:
        print(f"  Solved {idx}/{len(TARGETS)-1} remaining waypoints...")

# 2. Convert the full DH trajectory (approach + dense path) to CoppeliaSim joint space
ik_configs_sim = [dh_to_sim(q) for q in ik_configs_dh]

# 3. Resample the entire combined trajectory to match our desired movement duration
N_STEPS = int(TARGET_DUR / DT)
times   = [i * DT for i in range(N_STEPS)]
configs_sim = resample_to_n(ik_configs_sim, N_STEPS)

# 4. Send the single continuous motion to CoppeliaSim
move_id = 'waypoint_path_smoothed'
print(f"\nSending full trajectory to CoppeliaSim (Duration: {TARGET_DUR}s)...")
dispatch(sim, configs_sim, times, move_id, gripper_vel=GRIPPER_VEL)

print("  Moving ...", end='', flush=True)
wait_for_movement(sim, move_id)
print("\r  Done.               ")

sim.stopSimulation()
print("Simulation stopped.")