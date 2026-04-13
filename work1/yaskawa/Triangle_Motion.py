"""
Yaskawa GP8 — IK Solver (Position Only) → CoppeliaSim
=======================================================
Solves IK for a target TCP position only (no orientation constraint),
then sends the resulting joint trajectory to CoppeliaSim.

It also able to calculate waypoints along a straight-line Cartesian path, solve IK at each, and send the whole trajectory to CoppeliaSim.

Usage
-----
Edit TARGET_POS at the top, then run with CoppeliaSim open:
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
    np.array([0.640, -0.1, 0.51507]),
    np.array([0.640,  0.1, 0.51507]),
    np.array([0.640,  0.0, 0.71507]),
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
pos_start, R_start = fk(q_dh_current)
ro0, pi0, ya0 = R_to_rpy(R_start)

move_count = 0

for idx, TARGET_POS in enumerate(TARGETS):

    print("\n" + "=" * 55)
    print(f"Moving to target {idx+1}: {TARGET_POS.tolist()}")

    # --- Read current state (IMPORTANT: update every loop) ---
    q_sim_current = np.array([sim.getJointPosition(h) for h in joint_handles])
    q_dh_current  = JOINT_SIGN * q_sim_current
    pos_start, _  = fk(q_dh_current)

    # --- Reachability check ---
    print("Checking reachability ...")
    q_check = inverse_kinematics_pos(TARGET_POS, q_dh_current)
    pos_check, _ = fk(q_check)
    fk_err = np.linalg.norm(TARGET_POS - pos_check)
    print(f"  FK check: {np.round(pos_check, 4).tolist()}  err={fk_err:.4f} m")

    # --- Build trajectory ---
    print(f"Solving IK path ({IK_STEPS} waypoints)...")
    ik_configs_dh = build_cartesian_trajectory(
        pos_start  = pos_start,
        pos_target = TARGET_POS,
        q_seed     = q_dh_current,
        n_ik       = IK_STEPS,
    )

    # --- Convert + resample ---
    ik_configs_sim = [dh_to_sim(q) for q in ik_configs_dh]
    configs_sim    = resample_to_n(ik_configs_sim, N_STEPS)

    # --- Send motion ---
    move_count += 1
    move_id = f'ik_pos_move_{move_count}'

    print(f"Sending motion {move_count} ...")
    dispatch(sim, configs_sim, times, move_id, gripper_vel=GRIPPER_VEL)

    print("  Moving ...", end='', flush=True)
    wait_for_movement(sim, move_id)
    print("\r  Done.")

sim.stopSimulation()
print("Simulation stopped.")