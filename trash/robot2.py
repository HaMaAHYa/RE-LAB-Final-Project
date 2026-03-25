"""
Yaskawa GP8 — Cartesian Space Control via CoppeliaSim ZMQ Remote API
=====================================================================
Control the robot by typing commands at the prompt:

  x y z roll pitch yaw            move TCP (metres + degrees)
  x y z roll pitch yaw gripper    move TCP + gripper (0=open, 1=close)
  joints q1 q2 q3 q4 q5 q6        move directly in joint space (degrees)
  home                             return to zero pose
  where                            print current TCP pose and joints
  triangle                         run triangle path and return to start
  calib N                          calibrate sign for joint N (1-6)
  q                                quit

Architecture
------------
  DH table + tool offset  →  T_tcp  (numeric FK)
                          →  Jacobian J (6x6, finite-difference)
                          →  IK solver (Damped Least Squares)
                          →  JOINT_SIGN map (DH <-> CoppeliaSim)
                          →  straight Cartesian trajectory  ->  CoppeliaSim

  ALL Cartesian moves interpolate position linearly in Cartesian space,
  solving IK at every waypoint.  Joint-space moves (home, joints, calib)
  use joint-space interpolation as before.

Joint sign map  (JOINT_SIGN)
----------------------------
  +1 = same direction,  -1 = flipped.
  Confirmed: j3 = -1.  Run  calib N  to check the rest.

Triangle path
-------------
  Fixed orientation : roll=-90  pitch=90  yaw=-90  (degrees)
  Vertices          : P1=(0.640,  0.00023, 0.2)
                      P2=(0.24,  -0.35,    0.2)
                      P3=(0.24,   0.35,    0.2)
  Full path         : start -> P1 -> P2 -> P3 -> P1 -> start
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
import math

Pi = math.pi

# =====================================================================
# 1.  DH TABLE  (Classical DH, Yaskawa GP8)
# =====================================================================
DH_PARAMS = np.array([
    [0.040,  -Pi/2,   0.330,  0.0  ],
    [0.345,   0.0,    0.0,   -Pi/2 ],
    [0.040,  -Pi/2,   0.0,    0.0  ],
    [0.0,     Pi/2,   0.340,  0.0  ],
    [0.0,    -Pi/2,   0.0,    0.0  ],
    [0.0,     0.0,    0.24133,0.0  ],
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
    T = np.eye(4)
    for i, (a, alpha, d, theta_off) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(a, alpha, d, q[i] + theta_off)
    T = T @ T_TOOL
    return T[:3, 3].copy(), T[:3, :3].copy()

_pos0, _ = fk(np.zeros(6))
print(f"FK at zero pose : {np.round(_pos0, 5).tolist()}")
print(f"  (target       : [0.64, 0.00023, 0.71507])")

# =====================================================================
# 3.  JACOBIAN
# =====================================================================
FD_EPS = 1e-7

def jacobian(q):
    J = np.zeros((6, 6))
    for i in range(6):
        q_p, q_m = q.copy(), q.copy()
        q_p[i] += FD_EPS
        q_m[i] -= FD_EPS
        p_p, _ = fk(q_p)
        p_m, _ = fk(q_m)
        J[:3, i] = (p_p - p_m) / (2 * FD_EPS)
    T_acc = np.eye(4)
    for i, (a, alpha, d, theta_off) in enumerate(DH_PARAMS):
        J[3:, i] = T_acc[:3, 2]
        T_acc = T_acc @ dh_matrix(a, alpha, d, q[i] + theta_off)
    return J

# =====================================================================
# 4.  JOINT SIGN MAP
# =====================================================================
JOINT_SIGN = np.array([+1, +1, -1, +1, -1, +1], dtype=float)

def dh_to_sim(q): return JOINT_SIGN * q
def sim_to_dh(q): return JOINT_SIGN * q

# =====================================================================
# 5.  MATH HELPERS
# =====================================================================
def rotation_error(R_curr, R_tgt):
    Re = R_curr.T @ R_tgt
    return 0.5 * np.array([Re[2,1]-Re[1,2], Re[0,2]-Re[2,0], Re[1,0]-Re[0,1]])

def rpy_to_R(roll, pitch, yaw):
    cr, sr  = np.cos(roll),  np.sin(roll)
    cp, sp_ = np.cos(pitch), np.sin(pitch)
    cy, sy  = np.cos(yaw),   np.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp_],[0,1,0],[-sp_,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx

def R_to_rpy(R):
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    if abs(np.cos(pitch)) < 1e-6:
        roll, yaw = 0.0, np.arctan2(-R[1,2], R[1,1])
    else:
        roll = np.arctan2(R[2,1], R[2,2])
        yaw  = np.arctan2(R[1,0], R[0,0])
    return roll, pitch, yaw

# =====================================================================
# 6.  IK SOLVER  (Damped Least Squares + multi-seed fallback)
# =====================================================================
def _ik_single(target_pos, target_R, q0,
               max_iter=200, tol=1e-4, alpha=0.5, lambda_=0.01):
    """One DLS IK attempt from a single seed. Returns (q, final_error)."""
    q = np.array(q0, dtype=float)
    err_norm = float('inf')
    for _ in range(max_iter):
        pos_curr, R_curr = fk(q)
        error    = np.hstack((target_pos - pos_curr,
                              rotation_error(R_curr, target_R)))
        err_norm = np.linalg.norm(error)
        if err_norm < tol:
            return q, err_norm
        J     = jacobian(q)
        A     = J @ J.T + lambda_**2 * np.eye(6)
        J_dls = J.T @ np.linalg.inv(A)
        q     = q + alpha * (J_dls @ error)
    return q, err_norm


# Diverse fixed seeds that cover common arm configurations
_IK_FIXED_SEEDS = [
    np.zeros(6),
    np.deg2rad([ 0,  45,  45,  0,  45, 0]),
    np.deg2rad([ 0,  45,  45,  0, -45, 0]),
    np.deg2rad([ 0, -45, -45,  0,  45, 0]),
    np.deg2rad([45,  45,  45,  0,  45, 0]),
    np.deg2rad([-45, 45,  45,  0,  45, 0]),
    np.deg2rad([ 0,  30,  60,  0, -90, 0]),
    np.deg2rad([ 0,  60,  30,  0, -90, 0]),
]


def inverse_kinematics(target_pos, target_R, initial_guess,
                       max_iter=200, tol=1e-4, alpha=0.5, lambda_=0.01):
    """
    DLS IK with multi-seed fallback.

    First tries `initial_guess` (the warm-start seed passed in, usually
    the previous waypoint solution).  If that does not converge within
    tolerance, retries from several diverse fixed seeds and returns
    whichever result has the lowest final error.  This prevents the
    solver from getting permanently stuck when the arm needs to move
    across a configuration-branch boundary.
    """
    # --- primary attempt (warm start) --------------------------------
    q_best, err_best = _ik_single(target_pos, target_R, initial_guess,
                                  max_iter, tol, alpha, lambda_)
    if err_best < tol:
        return q_best

    # --- fallback: try diverse seeds ---------------------------------
    for seed in _IK_FIXED_SEEDS:
        q, err = _ik_single(target_pos, target_R, seed,
                            max_iter, tol, alpha, lambda_)
        if err < err_best:
            q_best, err_best = q, err
        if err_best < tol:
            break

    if err_best >= tol:
        print(f"  IK did not converge  (best err={err_best:.4f})")
    return q_best

# =====================================================================
# 7.  TRAJECTORY HELPERS
# =====================================================================
def smooth_trajectory(q_start, q_end, n):
    """Joint-space smoothstep. Used ONLY for joint-space moves."""
    traj = []
    for i in range(n):
        s = i / (n - 1)
        s = 3*s**2 - 2*s**3
        traj.append((1 - s)*q_start + s*q_end)
    return traj


def resample_to_n(configs, n):
    """
    Resample a list of configs (any length >= 2) to exactly n points
    using linear interpolation between adjacent entries.
    This ensures CoppeliaSim always receives exactly N_STEPS points
    regardless of how many IK waypoints were solved.
    """
    src = len(configs)
    if src == n:
        return configs
    out = []
    for i in range(n):
        # fractional index into source list
        f   = i * (src - 1) / (n - 1)
        lo  = int(f)
        hi  = min(lo + 1, src - 1)
        t   = f - lo
        out.append((1.0 - t) * configs[lo] + t * configs[hi])
    return out

# =====================================================================
# 8.  COPPELIA CONNECTION
# =====================================================================
client = RemoteAPIClient()
sim    = client.getObject('sim')

TARGET_ARM   = '/yaskawa'
STR_SIGNAL   = TARGET_ARM + '_executedMovId'
_exec_mov_id = 'notReady'

joint_handles = [sim.getObject(f'{TARGET_ARM}/joint{i+1}') for i in range(6)]
gripperEF     = sim.getObject('/gripperEF')

dt         = sim.getSimulationTimeStep()
TARGET_DUR = 5.0
N_STEPS    = int(TARGET_DUR / dt)
times      = [i * dt for i in range(N_STEPS)]

def wait_for_movement(move_id, timeout=60.0):
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

def get_q_sim():
    return np.array([sim.getJointPosition(h) for h in joint_handles])

def get_q_dh():
    return sim_to_dh(get_q_sim())

def get_tcp_pos():
    return np.array(sim.getObjectPosition(gripperEF, sim.handle_world))


def _dispatch(configs_sim, move_id, gripper_vel=0.0):
    """
    Pack exactly N_STEPS sim-space configs and fire them at CoppeliaSim.
    configs_sim must already have length N_STEPS (use resample_to_n first).
    """
    assert len(configs_sim) == N_STEPS, \
        f"Expected {N_STEPS} configs, got {len(configs_sim)}"
    md = {
        'id': move_id, 'type': 'pts', 'times': times,
        'j1': [q[0] for q in configs_sim],
        'j2': [q[1] for q in configs_sim],
        'j3': [q[2] for q in configs_sim],
        'j4': [q[3] for q in configs_sim],
        'j5': [q[4] for q in configs_sim],
        'j6': [q[5] for q in configs_sim],
        'gripper': [gripper_vel] * N_STEPS,
    }
    sim.callScriptFunction(f'remoteApi_movementDataFunction@{TARGET_ARM}',
                           sim.scripttype_childscript, md)
    sim.callScriptFunction(f'remoteApi_executeMovement@{TARGET_ARM}',
                           sim.scripttype_childscript, move_id)


def send_motion(q_start_sim, q_end_sim, target_pos, move_id, gripper_vel=0.0):
    """Joint-space move. Used for home and joints commands only."""
    traj = smooth_trajectory(q_start_sim, q_end_sim, N_STEPS)
    _dispatch(traj, move_id, gripper_vel)
    print("  Moving ...", end='', flush=True)
    wait_for_movement(move_id)
    pos  = get_tcp_pos()
    dist = np.linalg.norm(target_pos - pos)
    print(f"\r  Done.  TCP -> x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}  err={dist:.4f} m")


def move_linear(target_pos, target_R, move_id, gripper_vel=0.0, ik_steps=40):
    """
    Move TCP in a STRAIGHT CARTESIAN LINE.

    1. Solve IK at ik_steps evenly-spaced Cartesian waypoints.
    2. Resample the resulting joint configs to exactly N_STEPS points
       so CoppeliaSim receives a full-length trajectory and does NOT
       stop early or freeze waiting for a signal that never arrives.
    3. Dispatch and wait.
    """
    q_start_dh       = get_q_dh()
    pos_start, _     = fk(q_start_dh)

    # --- Step 1: solve IK along the straight Cartesian line ----------
    # Pre-solve the TARGET with multi-seed search so we know which
    # configuration branch to stay on throughout the leg.
    q_target, err_target = _ik_single(target_pos, target_R, q_start_dh)
    if err_target >= 1e-4:
        for seed in _IK_FIXED_SEEDS:
            q_t, e_t = _ik_single(target_pos, target_R, seed)
            if e_t < err_target:
                q_target, err_target = q_t, e_t
            if err_target < 1e-4:
                break

    q_seed = q_start_dh.copy()
    ik_configs_sim = []                      # sim-space, length = ik_steps

    for i in range(ik_steps):
        t   = i / (ik_steps - 1)
        s   = 3*t**2 - 2*t**3              # smoothstep timing (path stays straight)
        pos = (1.0 - s) * pos_start + s * target_pos
        q_solved, err = _ik_single(pos, target_R, q_seed)
        # If warm-start struggles, blend toward q_target as a fallback seed
        if err >= 1e-3:
            blended_seed = (1.0 - t) * q_start_dh + t * q_target
            q_b, e_b = _ik_single(pos, target_R, blended_seed)
            if e_b < err:
                q_solved, err = q_b, e_b
        q_seed = q_solved
        ik_configs_sim.append(dh_to_sim(q_seed).copy())

    # --- Step 2: resample to exactly N_STEPS -------------------------
    configs_sim = resample_to_n(ik_configs_sim, N_STEPS)

    # --- Step 3: dispatch --------------------------------------------
    _dispatch(configs_sim, move_id, gripper_vel)
    print("  Moving ...", end='', flush=True)
    wait_for_movement(move_id)
    pos  = get_tcp_pos()
    dist = np.linalg.norm(target_pos - pos)
    print(f"\r  Done.  TCP -> x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}  err={dist:.4f} m")

# =====================================================================
# 9.  SIGN CALIBRATION
# =====================================================================
def calibrate_signs(joint_index):
    test_angle = math.pi / 4
    q_sim_test = np.zeros(6)
    q_sim_test[joint_index] = test_angle

    move_id = f'calib_j{joint_index+1}'
    traj    = smooth_trajectory(np.zeros(6), q_sim_test, N_STEPS)
    _dispatch(traj, move_id)
    wait_for_movement(move_id)

    actual = get_tcp_pos()
    q_pos  = np.zeros(6); q_pos[joint_index] = +test_angle
    q_neg  = np.zeros(6); q_neg[joint_index] = -test_angle
    p_pos, _ = fk(q_pos)
    p_neg, _ = fk(q_neg)
    e_pos = np.linalg.norm(actual - p_pos)
    e_neg = np.linalg.norm(actual - p_neg)
    sign  = +1 if e_pos < e_neg else -1

    print(f"\nJoint {joint_index+1} calibration:")
    print(f"  Actual TCP    : {np.round(actual, 4).tolist()}")
    print(f"  FK sign=+1    : {np.round(p_pos, 4).tolist()}  err={e_pos:.4f}")
    print(f"  FK sign=-1    : {np.round(p_neg, 4).tolist()}  err={e_neg:.4f}")
    print(f"  => JOINT_SIGN[{joint_index}] = {sign:+d}")
    return sign

# =====================================================================
# 10. TRIANGLE PATH
# =====================================================================
TRIANGLE_R = rpy_to_R(np.deg2rad(-90.0), np.deg2rad(90.0), np.deg2rad(-90.0))

TRIANGLE_VERTICES = [
    np.array([0.640,  0.00023, 0.2]),   # P1
    np.array([0.24,  -0.35,    0.2]),   # P2
    np.array([0.24,   0.35,    0.2]),   # P3
]

TRIANGLE_SUBSTEPS = 40   # IK waypoints per leg (resampled to N_STEPS before dispatch)

LEG_LABELS = [
    "start → P1",
    "P1    → P2",
    "P2    → P3",
    "P3    → P1",
    "P1    → start",
]


def _solve_leg(pos_a, pos_b, R_fixed, q_seed_dh, n_sub):
    """
    Linearly interpolate n_sub+1 Cartesian points from pos_a to pos_b,
    solve IK for each (seeded from previous), return list of DH configs.
    """
    configs = []
    q = q_seed_dh.copy()
    for k in range(n_sub + 1):
        t   = k / n_sub
        pos = (1.0 - t) * pos_a + t * pos_b
        q   = inverse_kinematics(pos, R_fixed, q)
        configs.append(q.copy())
    return configs


def run_triangle(move_count_ref):
    q_start_dh = get_q_dh()
    pos_start  = get_tcp_pos()
    print(f"\n  Triangle start : {np.round(pos_start, 4).tolist()}")
    print(f"  Orientation    : roll=-90  pitch=90  yaw=-90 (deg, fixed)\n")

    path = [pos_start] + TRIANGLE_VERTICES + [TRIANGLE_VERTICES[0], pos_start]

    print("  Pre-solving IK for all waypoints ...")
    all_legs = []
    q_seed = q_start_dh.copy()

    for leg_idx in range(len(path) - 1):
        print(f"    Leg {leg_idx+1}/5 : {LEG_LABELS[leg_idx]}")
        leg_configs = _solve_leg(
            path[leg_idx], path[leg_idx + 1],
            TRIANGLE_R, q_seed, TRIANGLE_SUBSTEPS
        )
        all_legs.append(leg_configs)
        q_seed = leg_configs[-1]

    print(f"  IK solved.  Executing {len(path)-1} legs ...\n")

    for leg_idx, leg_configs in enumerate(all_legs):
        target_pos = path[leg_idx + 1]
        print(f"  Leg {leg_idx+1}/5 [{LEG_LABELS[leg_idx]}]"
              f"  → {np.round(target_pos, 4).tolist()}")

        # Convert DH -> sim, then resample to exactly N_STEPS
        configs_sim = [dh_to_sim(q) for q in leg_configs]
        configs_sim = resample_to_n(configs_sim, N_STEPS)

        move_count_ref[0] += 1
        move_id = f'tri_leg{leg_idx+1}_{move_count_ref[0]}'

        _dispatch(configs_sim, move_id)
        print("  Moving ...", end='', flush=True)
        wait_for_movement(move_id)
        pos  = get_tcp_pos()
        dist = np.linalg.norm(target_pos - pos)
        print(f"\r  Done.  TCP -> x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}  err={dist:.4f} m")

    print("\n  Triangle complete.  Robot back at start position.")

# =====================================================================
# 11. MAIN LOOP
# =====================================================================
HELP_TEXT = """
Commands
--------
  x y z roll pitch yaw [gripper]   Cartesian move (m + deg, gripper 0=open 1=close)
                                    TCP moves in a STRAIGHT CARTESIAN LINE.
  joints q1 q2 q3 q4 q5 q6         Joint move (degrees, sim space)
  home                              Return to zero pose
  where                             Show current TCP pose + joint angles
  triangle                          Run triangle path (RPY -90 90 -90) and return
  calib N                           Calibrate sign for joint N (1-6)
  q                                 Quit
"""

if __name__ == '__main__':

    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_advancing_running:
        time.sleep(0.1)

    print("Simulation running.")
    print(HELP_TEXT)

    wait_for_movement('ready')

    move_count = 0

    while True:
        print("─" * 55)
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue
        tokens = raw.lower().split()

        if tokens[0] == 'q':
            break

        if tokens[0] == 'help':
            print(HELP_TEXT)
            continue

        if tokens[0] == 'where':
            pos  = get_tcp_pos()
            q_dh = get_q_dh()
            _, R = fk(q_dh)
            ro, pi_, ya = R_to_rpy(R)
            print(f"  TCP position (m)  : x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")
            print(f"  TCP RPY (deg)     : roll={np.degrees(ro):.2f}  "
                  f"pitch={np.degrees(pi_):.2f}  yaw={np.degrees(ya):.2f}")
            print(f"  Joints DH  (deg)  : {np.round(np.degrees(q_dh), 2).tolist()}")
            print(f"  Joints sim (deg)  : {np.round(np.degrees(get_q_sim()), 2).tolist()}")
            continue

        if tokens[0] == 'home':
            move_count += 1
            send_motion(get_q_sim(), np.zeros(6),
                        target_pos=get_tcp_pos(),
                        move_id=f'home_{move_count}')
            continue

        if tokens[0] == 'triangle':
            mc_ref = [move_count]
            run_triangle(mc_ref)
            move_count = mc_ref[0]
            continue

        if tokens[0] == 'calib':
            if len(tokens) == 2 and tokens[1].isdigit():
                idx = int(tokens[1]) - 1
                if 0 <= idx <= 5:
                    calibrate_signs(idx)
                    continue
            print("  Usage:  calib N   (N = 1 to 6)")
            continue

        if tokens[0] == 'joints':
            if len(tokens) == 7:
                try:
                    q_sim_tgt = np.deg2rad([float(v) for v in tokens[1:]])
                    move_count += 1
                    send_motion(get_q_sim(), q_sim_tgt,
                                target_pos=get_tcp_pos(),
                                move_id=f'jmove_{move_count}')
                except ValueError:
                    print("  Invalid values. Provide 6 numbers in degrees.")
            else:
                print("  Usage:  joints q1 q2 q3 q4 q5 q6  (degrees)")
            continue

        if len(tokens) in (6, 7):
            try:
                vals = [float(v) for v in tokens]
            except ValueError:
                print("  Invalid input. Type 'help' for usage.")
                continue

            x, y, z, roll_d, pitch_d, yaw_d = vals[:6]
            gripper_cmd = int(float(vals[6])) if len(vals) == 7 else 0
            gripper_vel = -0.04 if gripper_cmd == 1 else +0.04

            target_pos = np.array([x, y, z])
            target_R   = rpy_to_R(np.deg2rad(roll_d),
                                   np.deg2rad(pitch_d),
                                   np.deg2rad(yaw_d))

            print(f"  Target pos (m) : x={x:.4f}  y={y:.4f}  z={z:.4f}")
            print(f"  Target RPY(deg): roll={roll_d:.2f}  pitch={pitch_d:.2f}  yaw={yaw_d:.2f}")

            q_curr_dh = get_q_dh()
            print(f"  Current joints DH (deg): {np.round(np.degrees(q_curr_dh), 2).tolist()}")

            # Quick reachability check
            q_tgt_dh = inverse_kinematics(target_pos, target_R, q_curr_dh)
            achieved, _ = fk(q_tgt_dh)
            fk_err = np.linalg.norm(target_pos - achieved)
            print(f"  FK check: {np.round(achieved, 4).tolist()}  err={fk_err:.4f} m")
            if fk_err > 0.01:
                print("  WARNING: FK error > 1 cm. Target may be near singularity or unreachable.")

            move_count += 1
            move_linear(
                target_pos,
                target_R,
                move_id=f'linMove_{move_count}',
                gripper_vel=gripper_vel,
            )
        else:
            print("  Unknown command. Type 'help' for usage.")

    sim.stopSimulation()
    print("Simulation stopped.")