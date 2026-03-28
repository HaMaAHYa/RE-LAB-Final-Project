"""
Yaskawa GP8 Orientation-Only Control for CoppeliaSim
=====================================================
Controls ONLY the wrist orientation (roll, pitch, yaw) of the robot
while keeping the TCP position fixed.

Strategy
--------
1. Read the current joint angles from the simulation.
2. Run FK to get the current TCP position and orientation.
3. FREEZE the position → keep the wrist center where it is.
4. For every target orientation, solve ONLY joints 4-6 analytically
   (closed-form ZYZ wrist decomposition) — joints 1-3 never move.
5. Dispatch the joint trajectory to CoppeliaSim.

Input format
------------
Each entry in ORIENTATION_TARGETS is a 3-tuple of (roll, pitch, yaw)
in DEGREES for readability. They are converted to radians internally.

The helper build_orientation_trajectory() smoothly interpolates between
orientations using SLERP-style matrix blending and re-solves the wrist
IK at every step for a clean, artifact-free motion.
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math
import time

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

Pi = math.pi

DT          = 0.05    # Must match CoppeliaSim simulation timestep (seconds)
IK_STEPS    = 60      # Interpolation waypoints between orientations (smoothness)
MOVE_DUR    = 2.0     # Duration (s) for each orientation move
HOLD_DUR    = 0.5     # Pause (s) between consecutive moves

# Yaskawa GP8 DH Parameters [a, alpha, d, theta_offset]
DH_PARAMS = np.array([
    [0.040,  -Pi/2,   0.330,   0.0  ],
    [0.345,   0.0,    0.0,    -Pi/2 ],
    [0.040,  -Pi/2,   0.0,     0.0  ],
    [0.0,     Pi/2,   0.340,   0.0  ],
    [0.0,    -Pi/2,   0.0,     0.0  ],
    [0.0,     0.0,    0.24133, 0.0  ],
], dtype=float)

# TCP Tool Transform
T_TOOL = np.array([
    [1, 0, 0,  0.00007],
    [0, 1, 0, -0.00023],
    [0, 0, 1,  0.01867],
    [0, 0, 0,  1.0    ],
], dtype=float)

# Joint sign map: +1 = DH matches sim direction, -1 = flipped
JOINT_SIGN = np.array([+1, +1, -1, +1, -1, +1], dtype=float)

# Wrist offset: distance from wrist center to TCP along tool z-axis
D6_TCP = DH_PARAMS[5, 2] + T_TOOL[2, 3]   # d6 + tool z-extension

# =====================================================================
# 2. ORIENTATION TARGETS
#    ─────────────────────────────────────────────────────────────────
#    Each entry is (roll_deg, pitch_deg, yaw_deg).
#    The robot visits them in order, starting from its current pose.
#    The TCP position stays fixed throughout.
#
#    Angles are INTRINSIC XYZ Euler (roll → pitch → yaw applied in
#    the body frame), the most common robotics convention.
# =====================================================================

ORIENTATION_TARGETS_DEG = [
    (  0,    0,    0),   # Home / neutral (tool pointing straight down)
]


# =====================================================================
# 3. MATH HELPERS
# =====================================================================

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

def euler_xyz_to_rot(rx, ry, rz):
    """Build rotation matrix from intrinsic XYZ Euler angles (radians)."""
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)

def slerp_rot(R_start, R_end, t):
    """
    Smooth rotation interpolation between two rotation matrices.

    Uses the matrix exponential / logarithm on SO(3):
      R(t) = R_start @ expm( t * logm(R_start^T @ R_end) )

    Falls back to linear blend + SVD re-orthogonalisation when the
    rotation angle is very small (avoids numerical issues near identity).
    """
    R_rel = R_start.T @ R_end

    # Extract axis-angle from R_rel via its trace
    cos_angle = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(cos_angle)

    if angle < 1e-8:
        # Essentially the same orientation — linear blend is exact enough
        R_blend = (1.0 - t) * R_start + t * R_end
    else:
        # Rodrigues' formula: R_rel^t = I + sin(t*θ)/sin(θ) * K + (1-cos(t*θ))/(1-cos(θ)) * K²
        # where K is the skew-symmetric matrix of the rotation axis.
        # Equivalent and simpler: use the matrix power.
        skew = (R_rel - R_rel.T) / (2.0 * math.sin(angle))
        axis = np.array([skew[2,1], skew[0,2], skew[1,0]])
        ta   = t * angle
        c, s = math.cos(ta), math.sin(ta)
        K    = np.array([[ 0,       -axis[2],  axis[1]],
                          [ axis[2],  0,       -axis[0]],
                          [-axis[1],  axis[0],  0      ]])
        R_step = np.eye(3) + s * K + (1 - c) * (K @ K)
        R_blend = R_start @ R_step

    # Re-orthogonalise via SVD to eliminate floating-point drift
    U, _, Vt = np.linalg.svd(R_blend)
    R_out = U @ Vt
    if np.linalg.det(R_out) < 0:
        U[:, -1] *= -1
        R_out = U @ Vt
    return R_out


# =====================================================================
# 4. FORWARD KINEMATICS
# =====================================================================

def dh_matrix(a, alpha, d, theta):
    """Single DH link transformation matrix."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1.0 ],
    ])

def fk_full(q):
    """Full FK → 4×4 TCP pose for joint vector q (DH convention)."""
    T = np.eye(4)
    for i, (a, alpha, d, theta_off) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(a, alpha, d, q[i] + theta_off)
    return T @ T_TOOL

def fk_to_joint3(q):
    """
    FK for joints 1-3 only, ending at the wrist center frame.
    This gives R_03: the rotation delivered by the first 3 joints.
    """
    T = np.eye(4)
    for i in range(3):
        a, alpha, d, theta_off = DH_PARAMS[i]
        T = T @ dh_matrix(a, alpha, d, q[i] + theta_off)
    # Apply joint-4 DH at q4=0 to get to the wrist center frame origin
    a, alpha, d, theta_off = DH_PARAMS[3]
    T = T @ dh_matrix(a, alpha, d, theta_off)
    return T


# =====================================================================
# 5. WRIST IK  (analytical ZYZ decomposition — joints 4, 5, 6 only)
# =====================================================================

def solve_wrist_orientation(q_123, R_desired):
    """
    Closed-form IK for the spherical wrist (joints 4-6).

    Given that joints 1-3 are FIXED at q_123, and we want the TCP to
    have orientation R_desired, we need the wrist to supply:

        R_wrist = R_03^T @ R_desired

    The GP8 wrist has a ZYZ-like structure (from the DH alphas):

        R_wrist = Rz(q4) @ Ry(q5) @ Rz(q6)

    Closed-form extraction:
        q5 = atan2( sqrt(r13²+r23²), r33 )
        q4 = atan2( r23/sin(q5), r13/sin(q5) )
        q6 = atan2( r32/sin(q5), -r31/sin(q5) )

    Singularity (q5 ≈ 0 or π): joints 4 and 6 are redundant.
    We set q4 = 0 and absorb everything into q6.

    Parameters
    ----------
    q_123    : array-like (3,) — current joints 1-3 angles (DH, radians)
    R_desired: 3×3 ndarray   — desired TCP rotation matrix

    Returns
    -------
    q456 : ndarray (3,) — joint angles for joints 4, 5, 6 (DH, radians)
    """
    R_03    = fk_to_joint3(q_123)[:3, :3]
    R_wrist = R_03.T @ R_desired

    r13 = R_wrist[0, 2]
    r23 = R_wrist[1, 2]
    r33 = R_wrist[2, 2]
    r31 = R_wrist[2, 0]
    r32 = R_wrist[2, 1]

    sin_q5 = math.sqrt(max(r13**2 + r23**2, 0.0))
    SING_TOL = 1e-6

    if sin_q5 > SING_TOL:
        q5 = math.atan2(sin_q5, r33)
        q4 = math.atan2(r23 / sin_q5, r13 / sin_q5)
        q6 = math.atan2(r32 / sin_q5, -r31 / sin_q5)
    else:
        # Gimbal lock — q4 and q6 are coupled; set q4=0
        q5 = 0.0 if r33 > 0 else math.pi
        q4 = 0.0
        q6 = math.atan2(R_wrist[1, 0], R_wrist[0, 0])

    return np.array([q4, q5, q6])


# =====================================================================
# 6. ORIENTATION TRAJECTORY BUILDER
# =====================================================================

def s_curve(t):
    """5th-order polynomial: smooth acceleration and deceleration."""
    return 6*t**5 - 15*t**4 + 10*t**3

def build_orientation_trajectory(q_fixed_123, R_start, R_end, n_steps=IK_STEPS):
    """
    Generate a smooth wrist trajectory from R_start to R_end while
    keeping joints 1-3 at q_fixed_123 (position locked).

    Parameters
    ----------
    q_fixed_123 : array-like (3,) — joints 1-3, held constant
    R_start     : 3×3 ndarray    — starting TCP orientation
    R_end       : 3×3 ndarray    — target TCP orientation
    n_steps     : int            — number of interpolation steps

    Returns
    -------
    configs_dh : list of (6,) ndarrays in DH convention
    """
    configs = []
    q_fixed = np.asarray(q_fixed_123)

    for k in range(n_steps):
        t = k / max(n_steps - 1, 1)
        s = s_curve(t)                       # smooth timing profile

        R_interp = slerp_rot(R_start, R_end, s)
        q_456    = solve_wrist_orientation(q_fixed, R_interp)

        q_full = np.concatenate([q_fixed, q_456])
        configs.append(q_full)

    return configs

def resample_to_n(configs, n):
    """Linearly resample a list of joint arrays to exactly n points."""
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

def dh_to_sim(q):
    """Apply joint sign map to convert DH angles → simulation angles."""
    return JOINT_SIGN * q


# =====================================================================
# 7. COPPELIA HELPERS
# =====================================================================

TARGET_ARM   = '/yaskawa'
STR_SIGNAL   = TARGET_ARM + '_executedMovId'
_exec_mov_id = 'notReady'

def wait_for_movement(sim, move_id, timeout=60.0):
    """Block until CoppeliaSim confirms the movement is done."""
    global _exec_mov_id
    _exec_mov_id = 'notReady'
    t0 = time.time()
    while _exec_mov_id != move_id:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Movement '{move_id}' timed out after {timeout}s")
        s = sim.getStringSignal(STR_SIGNAL)
        if s is not None:
            _exec_mov_id = s.decode() if isinstance(s, bytes) else s
        time.sleep(0.05)

def dispatch(sim, configs_sim, times, move_id):
    """Pack and send a joint trajectory to the CoppeliaSim Lua script."""
    assert len(configs_sim) == len(times), "length mismatch"
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
        'gripper': [0.0] * len(times),
    }
    sim.callScriptFunction(
        f'remoteApi_movementDataFunction@{TARGET_ARM}',
        sim.scripttype_childscript, md)
    sim.callScriptFunction(
        f'remoteApi_executeMovement@{TARGET_ARM}',
        sim.scripttype_childscript, move_id)

def get_tcp_pose_str(T):
    """Format a 4×4 pose matrix as a readable string for logging."""
    p = T[:3, 3]
    R = T[:3, :3]
    # Extract roll/pitch/yaw from rotation matrix (intrinsic XYZ)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        rx = math.atan2( R[2,1], R[2,2])
        ry = math.atan2(-R[2,0], sy)
        rz = math.atan2( R[1,0], R[0,0])
    else:
        rx = math.atan2(-R[1,2], R[1,1])
        ry = math.atan2(-R[2,0], sy)
        rz = 0.0
    return (f"pos=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})  "
            f"rpy=({math.degrees(rx):.1f}°, {math.degrees(ry):.1f}°, {math.degrees(rz):.1f}°)")


# =====================================================================
# 8. MAIN
# =====================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Connect and read current robot state
    # ------------------------------------------------------------------
    print("─" * 60)
    print("  Yaskawa GP8 — Orientation-Only Control")
    print("─" * 60)
    print("Connecting to CoppeliaSim...")

    client = RemoteAPIClient()
    sim    = client.getObject('sim')
    joint_handles = [sim.getObject(f'{TARGET_ARM}/joint{i+1}') for i in range(6)]

    q_sim_current = np.array([sim.getJointPosition(h) for h in joint_handles])
    q_dh_current  = JOINT_SIGN * q_sim_current

    T_current = fk_full(q_dh_current)
    print(f"\nCurrent TCP: {get_tcp_pose_str(T_current)}")

    # ------------------------------------------------------------------
    # Freeze joints 1-3 — they never move in this script
    # ------------------------------------------------------------------
    q_fixed_123 = q_dh_current[:3].copy()
    print(f"\nPosition locked at TCP xyz = "
          f"({T_current[0,3]:.4f}, {T_current[1,3]:.4f}, {T_current[2,3]:.4f})")
    print(f"Joints 1-3 frozen: {np.round(np.degrees(q_fixed_123), 2).tolist()} deg")

    # ------------------------------------------------------------------
    # Pre-compute all orientation trajectories
    # ------------------------------------------------------------------
    print(f"\nPre-computing {len(ORIENTATION_TARGETS_DEG)} orientation trajectories...")

    R_current = T_current[:3, :3].copy()

    all_trajectories = []
    for i, (roll_d, pitch_d, yaw_d) in enumerate(ORIENTATION_TARGETS_DEG):
        rx = math.radians(roll_d)
        ry = math.radians(pitch_d)
        rz = math.radians(yaw_d)

        R_target = euler_xyz_to_rot(rx, ry, rz)

        configs_dh  = build_orientation_trajectory(
            q_fixed_123, R_current, R_target, n_steps=IK_STEPS)
        configs_sim = [dh_to_sim(q) for q in configs_dh]

        N_STEPS   = int(MOVE_DUR / DT)
        times     = [k * DT for k in range(N_STEPS)]
        configs_resampled = resample_to_n(configs_sim, N_STEPS)

        all_trajectories.append({
            'configs':  configs_resampled,
            'times':    times,
            'id':       f'ori_move_{i+1}',
            'label':    f"(roll={roll_d:+.0f}°, pitch={pitch_d:+.0f}°, yaw={yaw_d:+.0f}°)",
            'R_target': R_target,
        })

        # Next move starts from where this one ends
        R_current = R_target

        print(f"  [{i+1:2d}/{len(ORIENTATION_TARGETS_DEG)}] "
              f"roll={roll_d:+5.1f}° pitch={pitch_d:+5.1f}° yaw={yaw_d:+5.1f}° — OK")

    # ------------------------------------------------------------------
    # Start simulation
    # ------------------------------------------------------------------
    print("\nStarting simulation...")
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_advancing_running:
        time.sleep(0.1)
    print("Simulation running.\n")

    wait_for_movement(sim, 'ready')

    # ------------------------------------------------------------------
    # Execute trajectories one by one
    # ------------------------------------------------------------------
    print("─" * 60)
    print(f"  Executing {len(all_trajectories)} orientation moves")
    print(f"  Duration per move : {MOVE_DUR}s  |  Hold between : {HOLD_DUR}s")
    print("─" * 60)

    for i, traj in enumerate(all_trajectories):
        move_id = traj['id']
        label   = traj['label']

        print(f"\nMove {i+1}/{len(all_trajectories)}  →  {label}")
        dispatch(sim, traj['configs'], traj['times'], move_id)

        print(f"  Executing...", end='', flush=True)
        wait_for_movement(sim, move_id)
        print(f"\r  Done ✓  {label}")

        if HOLD_DUR > 0 and i < len(all_trajectories) - 1:
            time.sleep(HOLD_DUR)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("  All orientation moves complete.")

    # Verify final TCP position hasn't drifted
    q_sim_final = np.array([sim.getJointPosition(h) for h in joint_handles])
    q_dh_final  = JOINT_SIGN * q_sim_final
    T_final     = fk_full(q_dh_final)

    pos_drift = np.linalg.norm(T_final[:3, 3] - fk_full(q_dh_current)[:3, 3])
    print(f"  Final TCP: {get_tcp_pose_str(T_final)}")
    print(f"  Position drift: {pos_drift*1000:.3f} mm  (should be near zero)")
    print("─" * 60)

    sim.stopSimulation()
    print("Simulation stopped.")