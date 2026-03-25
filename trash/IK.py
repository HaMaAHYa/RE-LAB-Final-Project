"""
Yaskawa GP8 — Standalone IK Solver
====================================
Extracted from the CoppeliaSim ZMQ control script.
No simulator connection required.

Usage
-----
Edit TARGET_POS and TARGET_RPY_DEG at the bottom, then run:
    python ik_solver.py
"""

import numpy as np
import math

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
# 3.  JACOBIAN  (finite-difference linear + analytic angular)
# =====================================================================
FD_EPS = 1e-7

def jacobian(q):
    J = np.zeros((6, 6))
    # Linear velocity rows — finite difference
    for i in range(6):
        q_p, q_m = q.copy(), q.copy()
        q_p[i] += FD_EPS
        q_m[i] -= FD_EPS
        p_p, _ = fk(q_p)
        p_m, _ = fk(q_m)
        J[:3, i] = (p_p - p_m) / (2 * FD_EPS)
    # Angular velocity rows — z-axis of each joint frame
    T_acc = np.eye(4)
    for i, (a, alpha, d, theta_off) in enumerate(DH_PARAMS):
        J[3:, i] = T_acc[:3, 2]
        T_acc = T_acc @ dh_matrix(a, alpha, d, q[i] + theta_off)
    return J

# =====================================================================
# 4.  MATH HELPERS
# =====================================================================
def rotation_error(R_curr, R_tgt):
    """Orientation error as a 3-vector (axis-angle residual)."""
    Re = R_curr.T @ R_tgt
    return 0.5 * np.array([Re[2,1]-Re[1,2],
                            Re[0,2]-Re[2,0],
                            Re[1,0]-Re[0,1]])

def rpy_to_R(roll, pitch, yaw):
    """Euler ZYX → rotation matrix."""
    cr, sr   = np.cos(roll),  np.sin(roll)
    cp, sp_  = np.cos(pitch), np.sin(pitch)
    cy, sy   = np.cos(yaw),   np.sin(yaw)
    Rx = np.array([[1,  0,   0 ], [0, cr, -sr], [0,  sr,  cr]])
    Ry = np.array([[cp, 0,  sp_], [0,  1,   0], [-sp_, 0, cp]])
    Rz = np.array([[cy,-sy,  0 ], [sy, cy,  0], [0,   0,   1]])
    return Rz @ Ry @ Rx

def R_to_rpy(R):
    """Rotation matrix → roll, pitch, yaw (radians)."""
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    if abs(np.cos(pitch)) < 1e-6:
        roll = 0.0
        yaw  = np.arctan2(-R[1,2], R[1,1])
    else:
        roll = np.arctan2(R[2,1], R[2,2])
        yaw  = np.arctan2(R[1,0], R[0,0])
    return roll, pitch, yaw

# =====================================================================
# 5.  IK SOLVER  (Damped Least Squares)
# =====================================================================
def inverse_kinematics(target_pos, target_R, initial_guess,
                        max_iter=200, tol=1e-4, alpha=0.5, lambda_=0.01):
    """
    Damped Least Squares IK.

    Parameters
    ----------
    target_pos    : array (3,)   desired TCP position in metres
    target_R      : array (3,3)  desired TCP rotation matrix
    initial_guess : array (6,)   starting joint angles in radians
    max_iter      : int          maximum iterations
    tol           : float        convergence tolerance (error norm)
    alpha         : float        step size
    lambda_       : float        damping factor

    Returns
    -------
    q : array (6,)  joint angles in radians
    """
    q = np.array(initial_guess, dtype=float)
    for _ in range(max_iter):
        pos_curr, R_curr = fk(q)
        error = np.hstack((target_pos - pos_curr,
                           rotation_error(R_curr, target_R)))
        if np.linalg.norm(error) < tol:
            break
        J     = jacobian(q)
        A     = J @ J.T + lambda_**2 * np.eye(6)
        J_dls = J.T @ np.linalg.inv(A)
        q     = q + alpha * (J_dls @ error)
    return q

# =====================================================================
# 6.  MAIN — edit TARGET_POS and TARGET_RPY_DEG to solve your IK
# =====================================================================
if __name__ == '__main__':

    # --- Target TCP pose -------------------------------------------
    TARGET_POS     = np.array([0.640,  0.00023, 0.2])   # metres  (x, y, z)
    TARGET_RPY_DEG = np.array([-90.0,  90.0,   -90.0])  # degrees (roll, pitch, yaw)

    # --- Initial joint guess (radians) ------------------------------
    INITIAL_GUESS  = np.zeros(6)

    # ----------------------------------------------------------------
    target_R = rpy_to_R(*np.deg2rad(TARGET_RPY_DEG))

    print("=" * 52)
    print("  Yaskawa GP8 — Standalone IK Solver")
    print("=" * 52)
    print(f"\nTarget position  (m)  : {TARGET_POS.tolist()}")
    print(f"Target RPY       (deg): {TARGET_RPY_DEG.tolist()}")
    print(f"Initial guess    (deg): {np.round(np.degrees(INITIAL_GUESS), 2).tolist()}")

    # FK at initial guess
    pos0, R0 = fk(INITIAL_GUESS)
    print(f"\nFK at initial guess   : {np.round(pos0, 5).tolist()}")

    # Solve IK
    q_sol = inverse_kinematics(target_R=target_R,
                               target_pos=TARGET_POS,
                               initial_guess=INITIAL_GUESS)

    # Verify with FK
    pos_achieved, R_achieved = fk(q_sol)
    pos_err = np.linalg.norm(TARGET_POS - pos_achieved)
    ro, pi_, ya = R_to_rpy(R_achieved)

    print("\n--- IK Solution ---")
    print(f"  Joint angles (deg) : {np.round(np.degrees(q_sol), 4).tolist()}")
    print(f"  Joint angles (rad) : {np.round(q_sol, 6).tolist()}")

    print("\n--- FK Verification ---")
    print(f"  Achieved pos  (m)  : {np.round(pos_achieved, 5).tolist()}")
    print(f"  Target   pos  (m)  : {TARGET_POS.tolist()}")
    print(f"  Position error (m) : {pos_err:.6f}")
    print(f"  Achieved RPY (deg) : roll={np.degrees(ro):.2f}  "
          f"pitch={np.degrees(pi_):.2f}  yaw={np.degrees(ya):.2f}")

    if pos_err < 1e-3:
        print("\n  ✓ IK converged successfully.")
    else:
        print(f"\n  ✗ WARNING: position error = {pos_err:.4f} m  "
              f"(target may be near singularity or out of reach).")