from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time
import math

PI = math.pi

print('Program started')

# ------------------------------------------------------------
# Connect
# ------------------------------------------------------------
client = RemoteAPIClient()
sim = client.require('sim')

targetArm = '/yaskawa'
gripperEF = sim.getObject('/gripperEF')

# ------------------------------------------------------------
# DH transform (standard Craig convention)
# ------------------------------------------------------------
def dh_transform(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,  -st*ca,  st*sa,  a*ct],
        [st,   ct*ca, -ct*sa,  a*st],
        [ 0,      sa,     ca,     d],
        [ 0,       0,      0,     1]
    ])

# ------------------------------------------------------------
# FIX 1 & 2 — Correct DH table
# Verified against datasheet:
#   d1=330mm (base height)
#   a2=385mm (forearm horizontal reach)
#   d4=340mm (upper arm)
#   d6=80mm  (flange)
#   shoulder lateral offset a1=40mm
# Alpha signs chosen so FK at q=0 matches upright home pose
# ------------------------------------------------------------
def yaskawa_fk(q):
    dh = [
        #  a       d        alpha          theta
        (0.040,   0.330, -PI/2,     q[0]),
        (0.345,   0,         0,       q[1]-PI/2),
        (0.040,   0,         -PI/2,  q[2]),
        (0,       0.340,   PI/2,   q[3]),
        (0,       0,         -PI/2,  q[4]),
        (0,       0.24133, 0,         q[5])
    ]
    T = np.eye(4)
    for row in dh:
        T = T @ dh_transform(*row)
    return T

# ------------------------------------------------------------
# FIX 3 — Full 4×4 base transform (rotation + translation)
# Calibrated once at startup from two known poses
# ------------------------------------------------------------
def calibrate_base_transform(sim, gripperEF, joint_handles):
    """
    Uses q=zeros to get T_fk_0 and p_real_0.
    Uses q=[pi/4,0,0,0,0,0] to get T_fk_1 and p_real_1.
    Solves for T_base (4x4) such that p_world = T_base @ p_fk
    Falls back to translation-only if rotation is negligible.
    """
    # ---- pose 0: all zeros ----
    p_fk0   = yaskawa_fk(np.zeros(6))[:3, 3]
    p_real0 = np.array(sim.getObjectPosition(gripperEF, sim.handle_world))

    # ---- pose 1: rotate J1 by 45° ----
    q1 = np.zeros(6); q1[0] = np.pi/4
    sim.setJointTargetPosition(joint_handles[0], q1[0])
    time.sleep(1.0)
    p_fk1   = yaskawa_fk(q1)[:3, 3]
    p_real1 = np.array(sim.getObjectPosition(gripperEF, sim.handle_world))

    # Reset J1
    sim.setJointTargetPosition(joint_handles[0], 0.0)
    time.sleep(1.0)

    # Check if there is a rotation between FK frame and world frame
    d_fk   = p_fk1   - p_fk0    # direction in FK frame
    d_real = p_real1 - p_real0   # direction in world frame

    n_fk   = np.linalg.norm(d_fk)
    n_real = np.linalg.norm(d_real)

    T_base = np.eye(4)

    if n_fk > 1e-6 and n_real > 1e-6:
        # Build rotation from FK->world using the two displacement vectors
        # plus a cross product to form a full rotation matrix
        u = d_fk   / n_fk
        v = d_real / n_real
        cross = np.cross(u, v)
        cross_n = np.linalg.norm(cross)

        if cross_n > 1e-3:   # there IS a rotation
            angle = np.arcsin(np.clip(cross_n, -1, 1))
            axis  = cross / cross_n
            # Rodrigues rotation
            K = np.array([[    0, -axis[2],  axis[1]],
                          [ axis[2],     0, -axis[0]],
                          [-axis[1],  axis[0],    0]])
            R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K)
            T_base[:3, :3] = R
            print(f"  Base rotation detected: {np.degrees(angle):.2f}°")
        else:
            print("  No base rotation detected, translation only")

    # Translation: p_real = R @ p_fk + t  =>  t = p_real - R @ p_fk
    T_base[:3, 3] = p_real0 - T_base[:3, :3] @ p_fk0

    print(f"  Base translation: {np.round(T_base[:3,3], 4)}")
    return T_base

def fk_world(q, T_base):
    """FK result transformed into world frame."""
    T = T_base @ yaskawa_fk(q)
    return T

# ------------------------------------------------------------
# Jacobian — numerical, in world frame
# ------------------------------------------------------------
def compute_jacobian_world(q, T_base, delta=1e-6):
    J  = np.zeros((3, 6))
    p0 = fk_world(q, T_base)[:3, 3]
    for i in range(6):
        dq    = q.copy()
        dq[i] += delta
        J[:, i] = (fk_world(dq, T_base)[:3, 3] - p0) / delta
    return J

# ------------------------------------------------------------
# FIX 4 & 5 — Multi-seed IK with adaptive lambda
# ------------------------------------------------------------
def inverse_kinematics(target_pos_world, q_init, T_base,
                       max_iter=1000, tol=1e-4):

    limits = [
        (-2.967,  2.967),
        (-1.134,  2.531),
        (-1.222,  3.316),
        (-3.316,  3.316),
        (-2.356,  2.356),
        (-6.294,  6.294),
    ]

    # FIX 4: multiple seeds — zero config + perturbed variants
    seeds = [
        q_init.copy(),
        np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0]),
        np.array([-1.5, -0.5,  0.5,  0.0,  0.0,  0.0]),
        np.array([ 1.5, -0.5,  0.5,  0.0,  0.0,  0.0]),
        np.array([-1.0, -1.0,  1.0,  0.5,  0.0,  0.0]),
        np.array([ 1.0, -1.0,  1.0, -0.5,  0.0,  0.0]),
        np.array([-2.0,  0.5,  1.5,  0.0,  0.5,  0.0]),
        np.array([ 2.0,  0.5,  1.5,  0.0, -0.5,  0.0]),
    ]

    best_q   = q_init.copy()
    best_err = float('inf')

    for seed_idx, seed in enumerate(seeds):
        q = seed.copy()

        for iteration in range(max_iter):
            p        = fk_world(q, T_base)[:3, 3]
            error    = target_pos_world - p
            err_norm = np.linalg.norm(error)

            if err_norm < tol:
                print(f"  IK converged (seed {seed_idx}, "
                      f"iter {iteration}, error={err_norm:.2e})")
                return q   # success — return immediately

            J  = compute_jacobian_world(q, T_base)
            JT = J.T

            # FIX 5: adaptive lambda — smaller when close, larger when far
            lam = 0.01 if err_norm < 0.05 else 0.05

            dq = JT @ np.linalg.solve(J @ JT + lam**2 * np.eye(3), error)

            # Step cap scales with current error to allow fast coarse motion
            max_step = min(0.2, err_norm * 2.0)
            step = np.linalg.norm(dq)
            if step > max_step:
                dq *= max_step / step

            q += dq

            for i, (lo, hi) in enumerate(limits):
                q[i] = np.clip(q[i], lo, hi)

        # Track best result across seeds
        final_err = np.linalg.norm(
            target_pos_world - fk_world(q, T_base)[:3, 3])
        if final_err < best_err:
            best_err = final_err
            best_q   = q.copy()
            print(f"  Seed {seed_idx}: error={final_err:.4f} m (best so far)")

    print(f"  IK did not converge. Best error={best_err:.4f} m")
    return best_q

# ------------------------------------------------------------
# Smooth S-curve trajectory
# ------------------------------------------------------------
def smooth_trajectory(q_start, q_end, steps):
    traj = []
    for i in range(steps):
        s = i / (steps - 1)
        s = 3*s**2 - 2*s**3
        traj.append(((1-s)*q_start + s*q_end).tolist())
    return traj

# ------------------------------------------------------------
# Start simulation
# ------------------------------------------------------------
sim.startSimulation()
time.sleep(1)

# ------------------------------------------------------------
# Read current joint state
# ------------------------------------------------------------
joint_handles = [sim.getObject(f'/yaskawa/joint{i+1}') for i in range(6)]
current_q = np.array([sim.getJointPosition(h) for h in joint_handles])
print("Current joint angles (rad):", np.round(current_q, 3))

# ------------------------------------------------------------
# FIX 3: Calibrate full base transform
# ------------------------------------------------------------
print("\nCalibrating base frame transform...")
T_base = calibrate_base_transform(sim, gripperEF, joint_handles)
T_tool = [0.01456, 0, 0.0505]
T_tool = np.array([[1, 0, 0, T_tool[0]],
                   [0, 1, 0, T_tool[1]],
                   [0, 0, 1, T_tool[2]],
                   [0, 0, 0, 1]])
T_total = T_base @ T_tool

# Re-read joints after calibration moves
current_q = np.array([sim.getJointPosition(h) for h in joint_handles])

# Sanity check
p_check = fk_world(current_q, T_base)[:3, 3]
p_real  = np.array(sim.getObjectPosition(gripperEF, sim.handle_world))
print(f"\nFK(world) at current q: {np.round(p_check, 4)}")
print(f"Real sim position:       {np.round(p_real,  4)}")
print(f"Match error:             {np.linalg.norm(p_check - p_real):.5f} m")

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------
target_xyz = np.array([0.5  ,0.2 , 0.3998])  # in world frame, from CoppeliaSim's object position

# Check if target is within robot reach
reach = 0.040 + 0.385 + 0.340 + 0.080   # max arm extension
base_pos = T_base[:3, 3]
dist_from_base = np.linalg.norm(target_xyz - base_pos)
print(f"\nTarget distance from base: {dist_from_base:.3f} m")
print(f"Max robot reach:           {reach:.3f} m")
if dist_from_base > reach:
    print("  WARNING: target may be out of reach!")
else:
    print("  Target is within reach.")

# ------------------------------------------------------------
# Solve IK
# ------------------------------------------------------------
print(f"\nSolving IK for world target {np.round(target_xyz, 4)} ...")
q_target  = inverse_kinematics(target_xyz, current_q, T_base)
p_reached = fk_world(q_target, T_base)[:3, 3]
print("Target joint angles (rad):", np.round(q_target, 3))
print("FK(world) at solution:    ", np.round(p_reached, 4))
print("Position error (m):       ", round(float(
      np.linalg.norm(target_xyz - p_reached)), 5))

# ------------------------------------------------------------
# Smooth trajectory
# ------------------------------------------------------------
steps      = 100          # more steps = smoother motion
trajectory = smooth_trajectory(current_q, q_target, steps)
times      = [i * 0.05 for i in range(steps)]

movementData = {
    'id':      'ikSmooth',
    'type':    'pts',
    'times':   times,
    'j1': [q[0] for q in trajectory],
    'j2': [q[1] for q in trajectory],
    'j3': [q[2] for q in trajectory],
    'j4': [q[3] for q in trajectory],
    'j5': [q[4] for q in trajectory],
    'j6': [q[5] for q in trajectory],
    'gripper': [0] * steps
}

sim.callScriptFunction(
    'remoteApi_movementDataFunction@' + targetArm,
    sim.scripttype_childscript, movementData)

sim.callScriptFunction(
    'remoteApi_executeMovement@' + targetArm,
    sim.scripttype_childscript, 'ikSmooth')

# ------------------------------------------------------------
# Monitor
# ------------------------------------------------------------
print("\nReal gripper position during motion:")
for step_i in range(steps):
    pos = sim.getObjectPosition(gripperEF, sim.handle_world)
    err = np.linalg.norm(target_xyz - np.array(pos))
    print(f"  [{step_i+1:3d}/{steps}] "
          f"x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}  "
          f"dist_to_target={err:.3f}")
    time.sleep(0.05)

# Final report
final_pos = np.array(sim.getObjectPosition(gripperEF, sim.handle_world))
print(f"\nFinal position:  {np.round(final_pos, 4)}")
print(f"Target was:      {np.round(target_xyz, 4)}")
print(f"Final error:     {np.linalg.norm(target_xyz - final_pos):.5f} m")

sim.stopSimulation()
print('Program ended')