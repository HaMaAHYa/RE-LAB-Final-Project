"""
Yaskawa GP8 Pick-and-Place Script for CoppeliaSim
-------------------------------------------------
Features:
1. Grabs object following a 3D Cartesian path (Position IK).
2. Moves to PLACE_POSITIONS (Position IK - Wrist Locked).
3. Rotates to PLACE_ORIENTATIONS by controlling ONLY the wrist (Joints 4, 5, 6) 
   using pure 3D rotation matrices (No 6x6 math).
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math
import time

# =====================================================================
# 1. USER CONFIGURATION & SETUP
# =====================================================================

Pi = math.pi

DT = 0.05               
IK_STEPS = 40           
GRIPPER_VEL = 0.0       
PLACE_DUR = 3.0         
ORIENT_DUR = 2.0        # Duration for the orientation movement

# Yaskawa GP8 Classical DH Parameters [a, alpha, d, theta_offset]
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

JOINT_SIGN = np.array([+1, +1, -1, +1, -1, +1], dtype=float)

# ---------------- Waypoints ----------------

TARGETS = [
    np.array([-0.032915, -0.887067, 0.486025]),
    np.array([-0.03231, -0.882233, 0.48512]),
    # ... (Truncated for readability, keeping start and end bounds of your original array)
    np.array([0.546247, -0.501349, 0.766561]),
]

TARGET_DUR = [i * DT for i in range(len(TARGETS))][-1] + 1.5

# Step 1: The robot will move to these XYZ positions
PLACE_POSITIONS = [
    np.array([0.250, -0.400, 0.6000]),   # Place 1
    np.array([0.015,  0.500, 0.7000]),   # Place 2
]

# Step 2: The robot will then rotate its wrist (J4, J5, J6) to these 3D orientations
# Format: [Roll, Pitch, Yaw] in degrees
PLACE_ORIENTATIONS = [
    np.array([0, 90, 0]),       # Turn to this orientation at Place 1
    np.array([-90, 45, -90]),   # Turn to this orientation at Place 2
]


# =====================================================================
# 2. ROBOT KINEMATICS (MATH)
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
    """Forward Kinematics to the TCP."""
    T = np.eye(4)
    for i, (a, alpha, d, theta_off) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(a, alpha, d, q[i] + theta_off)
    T = T @ T_TOOL
    return T[:3, 3].copy(), T[:3, :3].copy()

def jacobian_pos(q, fd_eps=1e-7):
    """3x6 Positional Jacobian."""
    Jv = np.zeros((3, 6))
    for i in range(6):
        q_p, q_m = q.copy(), q.copy()
        q_p[i] += fd_eps
        q_m[i] -= fd_eps
        p_p, _ = fk(q_p)
        p_m, _ = fk(q_m)
        Jv[:, i] = (p_p - p_m) / (2 * fd_eps)
    return Jv

# --- MODIFIED: Added lock_wrist parameter ---
def inverse_kinematics_pos(target_pos, initial_guess, lock_wrist=False, max_iter=300, tol=1e-4, alpha=0.5, lambda_=0.01):
    """Inverse Kinematics (IK) - Position Only."""
    q = np.array(initial_guess, dtype=float)
    for _ in range(max_iter):
        pos_curr, _ = fk(q)
        error = target_pos - pos_curr
        if np.linalg.norm(error) < tol:
            break
        
        Jv = jacobian_pos(q)
        
        # If wrist is locked, zero out the influence of joints 4, 5, and 6
        if lock_wrist:
            Jv[:, 3:] = 0.0
            
        A     = Jv @ Jv.T + lambda_**2 * np.eye(3) 
        J_dls = Jv.T @ np.linalg.inv(A)       
        dq    = alpha * (J_dls @ error)
        
        # Strictly enforce no movement on wrist joints
        if lock_wrist:
            dq[3:] = 0.0
            
        q = q + dq
    return q

# --- 3D Orientation Helpers ---
def rpy_to_R(roll, pitch, yaw):
    """Builds a 3x3 Rotation Matrix from Roll, Pitch, Yaw."""
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def fk_to_joint3(q):
    """Forward Kinematics to the Wrist Center (Uses 4x4 matrix)."""
    T = np.eye(4)
    for i in range(3):
        a, alpha, d, theta_off = DH_PARAMS[i]
        T = T @ dh_matrix(a, alpha, d, q[i] + theta_off)
    a, alpha, d, theta_off = DH_PARAMS[3]
    T = T @ dh_matrix(a, alpha, d, theta_off)
    return T

def solve_wrist_orientation(q_123, R_desired):
    """
    Solves Joints 4, 5, and 6 using 3x3 Geometric Math.
    Requires Joints 1, 2, and 3 to be frozen.
    """
    R_03 = fk_to_joint3(np.pad(q_123, (0, 3)))[:3, :3]
    R_wrist = R_03.T @ R_desired  # 3x3 Matrix Multiplication

    r13, r23, r33 = R_wrist[0, 2], R_wrist[1, 2], R_wrist[2, 2]
    r31, r32 = R_wrist[2, 0], R_wrist[2, 1]

    sin_q5 = math.sqrt(max(r13**2 + r23**2, 0.0))

    if sin_q5 > 1e-6:
        q5 = math.atan2(sin_q5, r33)
        q4 = math.atan2(r23 / sin_q5, r13 / sin_q5)
        q6 = math.atan2(r32 / sin_q5, -r31 / sin_q5)
    else:
        q5 = 0.0 if r33 > 0 else math.pi
        q4 = 0.0
        q6 = math.atan2(R_wrist[1, 0], R_wrist[0, 0])

    return np.array([q4, q5, q6])

def dh_to_sim(q):
    return JOINT_SIGN * q


# =====================================================================
# 3. TRAJECTORY GENERATION (PATH PLANNING)
# =====================================================================

def resample_to_n(configs, n):
    src = len(configs)
    if src == n: return configs
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
        t = k / max(n_steps - 1, 1)
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
    assert len(configs_sim) == len(times)
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
    sim.callScriptFunction(f'remoteApi_movementDataFunction@{TARGET_ARM}', sim.scripttype_childscript, md)
    sim.callScriptFunction(f'remoteApi_executeMovement@{TARGET_ARM}', sim.scripttype_childscript, move_id)


# =====================================================================
# 5. MAIN EXECUTION SCRIPT
# =====================================================================

if __name__ == "__main__":
    print("--- Connecting to CoppeliaSim ---")
    client = RemoteAPIClient()
    sim    = client.getObject('sim')
    joint_handles = [sim.getObject(f'{TARGET_ARM}/joint{i+1}') for i in range(6)]

    N_STEPS = int(TARGET_DUR / DT)
    times   = [i * DT for i in range(N_STEPS)]

    q_sim_current = np.array([sim.getJointPosition(h) for h in joint_handles])
    q_dh_current  = JOINT_SIGN * q_sim_current

    pos_start, _ = fk(q_dh_current)
    ik_configs_dh = [q_dh_current.copy()]
    q_seed = q_dh_current.copy()

    # --- Phase 1 & 2: Approach and Grab ---
    print("\n--- Phase 1 & 2: Calculating Cartesian approach and grab path ---")
    approach_configs = build_cartesian_trajectory(pos_start, TARGETS[0], q_seed, n_ik=IK_STEPS)
    ik_configs_dh.extend(approach_configs)
    q_seed = approach_configs[-1].copy()

    for idx in range(1, len(TARGETS)):
        target_pos = TARGETS[idx]
        q_ik = inverse_kinematics_pos(target_pos, q_seed)
        ik_configs_dh.append(q_ik.copy())
        q_seed = q_ik.copy()

    configs_sim_grab = resample_to_n([dh_to_sim(q) for q in ik_configs_dh], N_STEPS)


    # --- Phase 3: Calculate Placing & Wrist Orientation Trajectories ---
    print("\n--- Phase 3: Calculating Placing & Wrist Orientation Paths ---")
    q_current = ik_configs_dh[-1]  
    placing_tasks = []

    for i in range(len(PLACE_POSITIONS)):
        # 1. POSITION MOVE
        place_pos = PLACE_POSITIONS[i]
        print(f"  Calculating Position Move {i+1}...")
        
        # --- MODIFIED: Pass lock_wrist=True to keep J4, J5, J6 strictly frozen ---
        q_target_place = inverse_kinematics_pos(place_pos, q_current, lock_wrist=True)
        
        configs_dh_pos = build_joint_trajectory(q_current, q_target_place, n_steps=IK_STEPS)
        configs_sim_pos = [dh_to_sim(q) for q in configs_dh_pos]
        
        N_STEPS_PLACE = int(PLACE_DUR / DT)
        times_place   = [k * DT for k in range(N_STEPS_PLACE)]
        
        placing_tasks.append({
            'configs': resample_to_n(configs_sim_pos, N_STEPS_PLACE),
            'times': times_place,
            'id': f'move_place_{i+1}'
        })
        q_current = q_target_place 


        # 2. ORIENTATION MOVE (Wrist Only)
        target_rpy = np.radians(PLACE_ORIENTATIONS[i])
        print(f"  Calculating Wrist Orientation {i+1}...")
        
        # Build 3x3 Rotation matrix
        R_target = rpy_to_R(*target_rpy)
        
        # Freeze Joints 1, 2, and 3. Solve 4, 5, 6
        q_fixed_123 = q_current[:3]
        q_456_target = solve_wrist_orientation(q_fixed_123, R_target)
        
        # Combine them back into a full 6-joint array
        q_target_orient = np.concatenate([q_fixed_123, q_456_target])
        
        configs_dh_ori = build_joint_trajectory(q_current, q_target_orient, n_steps=IK_STEPS)
        configs_sim_ori = [dh_to_sim(q) for q in configs_dh_ori]
        
        N_STEPS_ORIENT = int(ORIENT_DUR / DT)
        times_orient   = [k * DT for k in range(N_STEPS_ORIENT)]
        
        placing_tasks.append({
            'configs': resample_to_n(configs_sim_ori, N_STEPS_ORIENT),
            'times': times_orient,
            'id': f'orient_place_{i+1}'
        })
        q_current = q_target_orient


    # --- Phase 4: Execute Simulation ---
    print("\n--- Starting Simulation ---")
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_advancing_running:
        time.sleep(0.1)
    print("Simulation running.") 
    wait_for_movement(sim, 'ready')

    print(f"\nSending Grabbing trajectory...")
    dispatch(sim, configs_sim_grab, times, 'waypoint_path_grabbing', gripper_vel=GRIPPER_VEL)
    wait_for_movement(sim, 'waypoint_path_grabbing')
    print("  Grab Done.")

    print(f"\n--- Executing Placing & Orientation Paths ---")
    for task in placing_tasks:
        print(f"Sending Task: {task['id']}...")
        dispatch(sim, task['configs'], task['times'], task['id'], gripper_vel=0.0)
        wait_for_movement(sim, task['id'])
        print(f"  {task['id']} Done.")
        time.sleep(0.2)

    sim.stopSimulation()
    print("\nSimulation stopped.")