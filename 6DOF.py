"""
Yaskawa GP8 — Interactive IK Controller for CoppeliaSim
=========================================================
Fixed: joint seed and FK pose now match what CoppeliaSim shows.

Key corrections vs previous version:
  1. q_dh is ALWAYS re-read from the simulator at the top of every loop
     iteration — never carried over from IK arithmetic, so drift is impossible.
  2. sim_to_dh / dh_to_sim are explicit named functions so the sign
     convention is applied in one place only (no accidental double-flip).
  3. Startup DIAGNOSTIC block prints raw sim joints + FK pose alongside
     CoppeliaSim's own TCP readout (gripperEF) so any offset is visible.
     Compare "Delta FK-sim" — it should be near zero. If not, adjust
     JOINT_SIGN or T_TOOL until it is.

Usage:
    python gp8_ik_controller.py
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math
import time

Pi = math.pi

# =====================================================================
# CONFIG
# =====================================================================
TARGET_ARM   = '/yaskawa'
STR_SIGNAL   = TARGET_ARM + '_executedMovId'

# DH-space = JOINT_SIGN * sim-space.
# Flip individual signs here if FK still disagrees with sim TCP readout.
JOINT_SIGN   = np.array([+1, +1, -1, +1, -1, +1], dtype=float)

IK_MAX_ITER  = 500
IK_TOL       = 1e-5
IK_ALPHA     = 0.5
IK_LAMBDA    = 0.01

SNAP_DUR     = 0.05   # seconds, must be >= simulation DT
DT           = 0.05

# =====================================================================
# DH TABLE  (Classical DH, Yaskawa GP8)
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
# FORWARD KINEMATICS
# =====================================================================
def dh_matrix(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,   sa,    ca,    d   ],
        [ 0,   0,     0,     1.0 ],
    ])

def fk(q):
    """Returns (pos [3] m, R [3x3])."""
    T = np.eye(4)
    for i, (a, alpha, d, theta_off) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(a, alpha, d, q[i] + theta_off)
    T = T @ T_TOOL
    return T[:3, 3].copy(), T[:3, :3].copy()

# =====================================================================
# JACOBIAN  (6x6 finite difference)
# =====================================================================
FD_EPS = 1e-7

def _rpy_from_R(R):
    """ZYX Euler -> (roll, pitch, yaw) radians."""
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    if abs(np.cos(pitch)) < 1e-6:
        roll = 0.0
        yaw  = np.arctan2(-R[1, 2], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])

def _fk_pose(q):
    pos, R = fk(q)
    return np.concatenate([pos, _rpy_from_R(R)])

def jacobian_full(q):
    """6x6 Jacobian via central finite differences."""
    J = np.zeros((6, 6))
    for i in range(6):
        qp, qm = q.copy(), q.copy()
        qp[i] += FD_EPS
        qm[i] -= FD_EPS
        J[:, i] = (_fk_pose(qp) - _fk_pose(qm)) / (2 * FD_EPS)
    return J

def jacobian_pos(q):
    """3x6 positional Jacobian only."""
    return jacobian_full(q)[:3, :]

# =====================================================================
# ROTATION HELPERS
# =====================================================================
def rpy_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy,  cy, 0], [0,   0,  1]])
    Ry = np.array([[cp,   0, sp], [ 0,   1, 0], [-sp, 0, cp]])
    Rx = np.array([[ 1,   0,  0], [ 0,  cr,-sr], [0,  sr, cr]])
    return Rz @ Ry @ Rx

def _angle_axis_error(R_curr, R_tgt):
    R_err = R_tgt @ R_curr.T
    trace = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
    angle = np.arccos(trace)
    if abs(angle) < 1e-9:
        return np.zeros(3)
    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]) / (2 * np.sin(angle))
    return axis * angle

# =====================================================================
# IK SOLVERS
# =====================================================================
def ik_position_only(target_pos, q0,
                     max_iter=IK_MAX_ITER, tol=IK_TOL,
                     alpha=IK_ALPHA, lam=IK_LAMBDA):
    q = q0.copy()
    for _ in range(max_iter):
        pos, _ = fk(q)
        err = target_pos - pos
        if np.linalg.norm(err) < tol:
            break
        Jv = jacobian_pos(q)
        A  = Jv @ Jv.T + lam**2 * np.eye(3)
        dq = Jv.T @ np.linalg.inv(A) @ err
        q += alpha * dq
    return q

def ik_full(target_pos, target_R, q0,
            max_iter=IK_MAX_ITER, tol=IK_TOL,
            alpha=IK_ALPHA, lam=IK_LAMBDA):
    q = q0.copy()
    for _ in range(max_iter):
        pos, R = fk(q)
        e_pos = target_pos - pos
        e_rot = _angle_axis_error(R, target_R)
        err   = np.concatenate([e_pos, e_rot])
        if np.linalg.norm(err) < tol:
            break
        J  = jacobian_full(q)
        A  = J @ J.T + lam**2 * np.eye(6)
        dq = J.T @ np.linalg.inv(A) @ err
        q += alpha * dq
    return q

# =====================================================================
# JOINT SPACE CONVERSIONS
# =====================================================================
def sim_to_dh(q_sim):
    """Simulator joint angles  ->  DH joint angles."""
    return JOINT_SIGN * q_sim

def dh_to_sim(q_dh):
    """DH joint angles  ->  simulator joint angles.
    JOINT_SIGN is its own inverse because each element is +1 or -1."""
    return JOINT_SIGN * q_dh

def read_joints(sim, handles):
    """Return (q_dh, q_sim) both as numpy arrays."""
    q_sim = np.array([sim.getJointPosition(h) for h in handles])
    return sim_to_dh(q_sim), q_sim

# =====================================================================
# COPPELIA MOTION HELPERS
# =====================================================================
_exec_mov_id = 'notReady'

def wait_for_movement(sim, move_id, timeout=30.0):
    global _exec_mov_id
    _exec_mov_id = 'notReady'
    t0 = time.time()
    while _exec_mov_id != move_id:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Movement '{move_id}' timed out after {timeout} s")
        s = sim.getStringSignal(STR_SIGNAL)
        if s is not None:
            _exec_mov_id = s.decode() if isinstance(s, bytes) else s
        time.sleep(0.02)

def dispatch(sim, configs_sim, times, move_id, gripper_vel=0.0):
    """Pack and send joint configs to the robot Lua script and execute."""
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
    sim.callScriptFunction(
        f'remoteApi_movementDataFunction@{TARGET_ARM}',
        sim.scripttype_childscript, md)
    sim.callScriptFunction(
        f'remoteApi_executeMovement@{TARGET_ARM}',
        sim.scripttype_childscript, move_id)

def snap_to(sim, q_sim_from, q_sim_to, move_id):
    """Instant 2-point snap move."""
    dispatch(sim, [q_sim_from, q_sim_to], [0.0, SNAP_DUR], move_id)
    wait_for_movement(sim, move_id)

# =====================================================================
# DISPLAY / DIAGNOSTIC HELPERS
# =====================================================================
def print_pose(label, pos, R):
    rpy = np.degrees(_rpy_from_R(R))
    print(f"  {label}")
    print(f"    X={pos[0]*1000:9.3f}  Y={pos[1]*1000:9.3f}  Z={pos[2]*1000:9.3f}  mm")
    print(f"    Roll={rpy[0]:8.3f}    Pitch={rpy[1]:8.3f}    Yaw={rpy[2]:8.3f}    deg")

def print_diagnostic(sim, handles):
    """
    Compare Python FK pose against CoppeliaSim's own TCP object.
    The 'Delta FK-sim' line should be close to zero.
    If X/Y/Z are off, a sign in JOINT_SIGN is wrong.
    If only orientation is off, T_TOOL may need adjustment.
    """
    q_dh, q_sim = read_joints(sim, handles)
    pos_fk, R_fk = fk(q_dh)
    rpy_fk = np.degrees(_rpy_from_R(R_fk))

    print("\n  ── STARTUP DIAGNOSTIC ──────────────────────────────────")
    print("  Joint angles:")
    print(f"  {'Joint':<6} {'Sim (deg)':>10} {'DH (deg)':>10} {'Sign':>6}")
    for i in range(6):
        print(f"  J{i+1:<5} {math.degrees(q_sim[i]):>10.3f} "
              f"{math.degrees(q_dh[i]):>10.3f} {int(JOINT_SIGN[i]):>+6d}")

    print(f"\n  Python FK  -> X={pos_fk[0]*1000:8.3f}  Y={pos_fk[1]*1000:8.3f}  "
          f"Z={pos_fk[2]*1000:8.3f}  mm")
    print(f"               Roll={rpy_fk[0]:7.3f}  Pitch={rpy_fk[1]:7.3f}  "
          f"Yaw={rpy_fk[2]:7.3f}  deg")

    # Try common TCP/tool object names in the scene
    for name in ['/gripperEF', f'{TARGET_ARM}/gripperEF',
                 f'{TARGET_ARM}/tool', '/tool', f'{TARGET_ARM}/TCP', '/TCP']:
        try:
            h   = sim.getObject(name)
            pos = sim.getObjectPosition(h, -1)
            ori = sim.getObjectOrientation(h, -1)   # Euler alpha/beta/gamma
            print(f"\n  CoppeliaSim '{name}'")
            print(f"               X={pos[0]*1000:8.3f}  Y={pos[1]*1000:8.3f}  "
                  f"Z={pos[2]*1000:8.3f}  mm")
            print(f"               Euler a={math.degrees(ori[0]):7.3f}  "
                  f"b={math.degrees(ori[1]):7.3f}  g={math.degrees(ori[2]):7.3f}  deg")
            dx = (pos_fk[0] - pos[0]) * 1000
            dy = (pos_fk[1] - pos[1]) * 1000
            dz = (pos_fk[2] - pos[2]) * 1000
            print(f"  Delta FK-sim:  dX={dx:+8.3f}  dY={dy:+8.3f}  dZ={dz:+8.3f}  mm")
            break
        except Exception:
            continue
    else:
        print("\n  (No TCP object found in scene for comparison)")

    print("  ────────────────────────────────────────────────────────\n")

# =====================================================================
# INTERACTIVE LOOP
# =====================================================================
def prompt_target(current_pos, current_R, mode):
    cur_rpy = np.degrees(_rpy_from_R(current_R))
    print()
    print("  Enter target (Enter = keep current, 'q' = quit):")

    def ask(label, cur):
        s = input(f"    {label} [{cur:.4f}]: ").strip()
        if s.lower() == 'q':
            raise SystemExit
        return float(s) if s else cur

    try:
        x = ask("X   (mm)", current_pos[0] * 1000) / 1000.0
        y = ask("Y   (mm)", current_pos[1] * 1000) / 1000.0
        z = ask("Z   (mm)", current_pos[2] * 1000) / 1000.0

        if mode == '6':
            roll  = math.radians(ask("Roll  (deg)", cur_rpy[0]))
            pitch = math.radians(ask("Pitch (deg)", cur_rpy[1]))
            yaw   = math.radians(ask("Yaw   (deg)", cur_rpy[2]))
            R_tgt = rpy_to_R(roll, pitch, yaw)
        else:
            R_tgt = None

    except (ValueError, EOFError):
        print("  Invalid input - keeping current pose.")
        return None, None

    return np.array([x, y, z]), R_tgt

# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 60)
    print("  Yaskawa GP8 - Interactive IK Controller (CoppeliaSim)")
    print("=" * 60)

    client = RemoteAPIClient()
    sim    = client.getObject('sim')
    print("  Connected to CoppeliaSim.\n")

    joint_handles = [sim.getObject(f'{TARGET_ARM}/joint{i+1}') for i in range(6)]

    # Start simulation if needed
    if sim.getSimulationState() != sim.simulation_advancing_running:
        sim.startSimulation()
        while sim.getSimulationState() != sim.simulation_advancing_running:
            time.sleep(0.05)
        print("  Simulation started.")

    # Wait for robot Lua script ready signal
    print("  Waiting for robot script ready signal ...")
    wait_for_movement(sim, 'ready')
    print("  Robot ready.")

    # ── STARTUP DIAGNOSTIC ───────────────────────────────────────────
    # If "Delta FK-sim" is large, adjust JOINT_SIGN above until it is
    # near zero before trusting any IK results.
    print_diagnostic(sim, joint_handles)
    # ─────────────────────────────────────────────────────────────────

    # IK mode
    print("  IK mode:")
    print("    3 - Position only  (orientation free)")
    print("    6 - Full 6-DOF     (position + orientation)")
    mode = input("  Choose [3/6, default=3]: ").strip() or '3'
    if mode not in ('3', '6'):
        mode = '3'
    print(f"  Mode: {'Position only' if mode == '3' else 'Full 6-DOF'}\n")
    print("  Type 'q' at any prompt to quit.\n")

    move_counter = 0

    while True:
        # Always re-read joints from sim — never carry over IK state
        q_dh, q_sim_cur = read_joints(sim, joint_handles)
        pos_curr, R_curr = fk(q_dh)
        print_pose("Current TCP:", pos_curr, R_curr)

        try:
            pos_tgt, R_tgt = prompt_target(pos_curr, R_curr, mode)
        except SystemExit:
            break

        if pos_tgt is None:
            continue

        # Solve IK
        print("  Solving IK ...", end=' ', flush=True)
        if mode == '6':
            q_new = ik_full(pos_tgt, R_tgt, q_dh)
        else:
            q_new = ik_position_only(pos_tgt, q_dh)
        print("done.")

        # Verify IK residual before sending
        pos_achieved, R_achieved = fk(q_new)
        pos_err = np.linalg.norm(pos_tgt - pos_achieved) * 1000  # mm
        if mode == '6':
            rot_err = np.degrees(np.linalg.norm(_angle_axis_error(R_achieved, R_tgt)))
            ok = pos_err < 1.0 and rot_err < 0.5
        else:
            rot_err = None
            ok = pos_err < 1.0

        print()
        print_pose("IK result TCP:", pos_achieved, R_achieved)
        print(f"  Position error : {pos_err:.3f} mm", end='')
        if rot_err is not None:
            print(f"    Orientation error : {rot_err:.3f} deg", end='')
        print()

        if not ok:
            print("  WARNING: Large IK residual - near singularity or outside workspace?")
            retry = input("  Send anyway? (y/N): ").strip().lower()
            if retry != 'y':
                print("  Skipped.\n")
                continue

        # Dispatch snap
        move_counter += 1
        move_id   = f'snap_{move_counter:04d}'
        q_sim_new = dh_to_sim(q_new)

        print(f"  Dispatching '{move_id}' ...", end=' ', flush=True)
        snap_to(sim, q_sim_cur, q_sim_new, move_id)
        print("done.")

        # Re-read sim after move to confirm
        q_dh_after, _ = read_joints(sim, joint_handles)
        pos_after, R_after = fk(q_dh_after)
        print_pose("  Post-move TCP:", pos_after, R_after)
        final_err = np.linalg.norm(pos_tgt - pos_after) * 1000
        print(f"  Final position error: {final_err:.3f} mm\n")

    sim.stopSimulation()
    print("\n  Simulation stopped. Bye.")


if __name__ == '__main__':
    main()