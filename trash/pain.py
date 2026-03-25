from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
import math

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

Pi = math.pi
JOINT_SIGN = np.array([+1, +1, -1, +1, -1, +1], dtype=float)

def dh_to_sim(q): return JOINT_SIGN * q
def sim_to_dh(q): return JOINT_SIGN * q

A = np.array([0.64106, 0.0,   0.71507])
B = np.array([0.64106, -0.1,  0.51507])
C = np.array([0.64106,  0.1,  0.51507])


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

def dh_matrix(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1.0 ],
    ])

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

    
def fk(q):
    T = np.eye(4)
    for i, (a, alpha, d, theta_off) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(a, alpha, d, q[i] + theta_off)
    T = T @ T_TOOL
    return T[:3, 3].copy(), T[:3, :3].copy()

def get_q_sim():
    return np.array([sim.getJointPosition(h) for h in joint_handles])

def get_tcp_pos():
    return np.array(sim.getObjectPosition(gripperEF, sim.handle_world))

FD_EPS = 1e-7

def get_q_dh():
    return sim_to_dh(get_q_sim())

def rpy_to_R(roll, pitch, yaw):
    cr, sr  = np.cos(roll),  np.sin(roll)
    cp, sp_ = np.cos(pitch), np.sin(pitch)
    cy, sy  = np.cos(yaw),   np.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp_],[0,1,0],[-sp_,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx

def rotation_error(R_curr, R_tgt):
    Re = R_curr.T @ R_tgt
    return 0.5 * np.array([Re[2,1]-Re[1,2], Re[0,2]-Re[2,0], Re[1,0]-Re[0,1]])


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

def inverse_kinematics(target_pos, target_R, initial_guess,
                       max_iter=200, tol=1e-4, alpha=0.5, lambda_=0.01):
    q = np.array(initial_guess, dtype=float)
    for i in range(max_iter):
        pos_curr, R_curr = fk(q)
        error = np.hstack((target_pos - pos_curr,
                           rotation_error(R_curr, target_R)))
        err_norm = np.linalg.norm(error)
        if err_norm < tol:
            return q
        J     = jacobian(q)
        A     = J @ J.T + lambda_**2 * np.eye(6)
        J_dls = J.T @ np.linalg.inv(A)
        q     = q + alpha * (J_dls @ error)
    print(f"  IK did not converge  (err={np.linalg.norm(error):.4f})")
    return q
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
    q_seed = q_start_dh.copy()
    ik_configs_sim = []                      # sim-space, length = ik_steps

    for i in range(ik_steps):
        t   = i / (ik_steps - 1)
        s   = 3*t**2 - 2*t**3              # smoothstep timing (path stays straight)
        pos = (1.0 - s) * pos_start + s * target_pos
        q_seed = inverse_kinematics(pos, target_R, q_seed)
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


def interpolate(P1, P2, N, include_last=False):
    pts = []
    for i in range(N):
        t = i / (N - 1)
        p = (1 - t) * P1 + t * P2
        pts.append(p)
    if not include_last:
        pts = pts[:-1]  # avoid duplicate corner
    return pts

N = 10

# Build triangle path
points = []
points += interpolate(A, B, N)
points += interpolate(B, C, N)
points += interpolate(C, A, N, include_last=True)  # close loop

sim.startSimulation()

while sim.getSimulationState() != sim.simulation_advancing_running:
    time.sleep(0.1)

wait_for_movement('ready')

move_count = 0



for i, p in enumerate(points):
    
    x, y, z, roll_d, pitch_d, yaw_d = p[0], p[1], p[2], -90.0, 90.0, -90.0
    
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
        gripper_vel=0.00,
    )
    time.sleep(0.1)