"""
Yaskawa GP8 Pick-and-Place Script for CoppeliaSim
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
import numpy as np
import math
import time

# =====================================================================
# 1. USER CONFIGURATION & SETUP
# =====================================================================
client = RemoteAPIClient()
sim    = client.getObject('sim')
Pi = math.pi

DT = 0.05
IK_STEPS = 40
G_CLOSE = -0.04
G_OPEN  =  0.04
G_HOLD  =  0.0
PLACE_DUR = 3.0
LIFT_DUR  = 3.0  # Duration for the final lift phase

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
WRIST_OFFSET = T_TOOL[2, 3]  # 0.01867 m

cup = sim.getObject('./Cup/cup_pose')
cup_pos = sim.getObjectPosition(cup, sim.handle_world)
print(cup_pos)
PLACE_POSITIONS = [
    np.array([cup_pos[0] - 0.15, cup_pos[1], cup_pos[2]]),
    np.array(cup_pos),
]

APPROACH_DUR = IK_STEPS * DT
TARGET_DUR   = APPROACH_DUR

# =====================================================================
# 2. DATA LOGGING
# =====================================================================
DATA_LOG = {
    't': [],
    'cup_pos': [], 'cup_vel': [], 'cup_ori': [], 'cup_omega': [],
    'ee_pos':  [], 'ee_vel':  [], 'ee_ori':  [], 'ee_omega':  [],
    'joint_pos': [], 'gripper_cmd': [],
}
_current_gripper_cmd  = G_HOLD
_dynamic_gripper_list = None
_dynamic_gripper_t0   = None

def log_simulation_data(sim, cup_handle, ee_handle):
    global _current_gripper_cmd, _dynamic_gripper_list, _dynamic_gripper_t0
    t = sim.getSimulationTime()
    if len(DATA_LOG['t']) > 0 and t <= DATA_LOG['t'][-1]:
        return
    if cup_handle != -1:
        cp = sim.getObjectPosition(cup_handle, sim.handle_world)
        cv, cw = sim.getObjectVelocity(cup_handle)
        co = sim.getObjectOrientation(cup_handle, sim.handle_world)
    else:
        cp, cv, cw, co = [0,0,0],[0,0,0],[0,0,0],[0,0,0]
    if ee_handle != -1:
        ep = sim.getObjectPosition(ee_handle, sim.handle_world)
        ev, ew = sim.getObjectVelocity(ee_handle)
        eo = sim.getObjectOrientation(ee_handle, sim.handle_world)
    else:
        ep, ev, ew, eo = [0,0,0],[0,0,0],[0,0,0],[0,0,0]
    if 'joint_handles' in globals() and joint_handles:
        jp = [sim.getJointPosition(h) for h in joint_handles]
    else:
        jp = [0.0] * 6
    if _dynamic_gripper_list is not None and _dynamic_gripper_t0 is not None:
        idx = max(0, min(int((t - _dynamic_gripper_t0) / DT), len(_dynamic_gripper_list)-1))
        gripper_val = _dynamic_gripper_list[idx]
    else:
        gripper_val = _current_gripper_cmd
    DATA_LOG['t'].append(t)
    DATA_LOG['cup_pos'].append(cp);  DATA_LOG['cup_vel'].append(cv)
    DATA_LOG['cup_ori'].append(co);  DATA_LOG['cup_omega'].append(cw)
    DATA_LOG['ee_pos'].append(ep);   DATA_LOG['ee_vel'].append(ev)
    DATA_LOG['ee_ori'].append(eo);   DATA_LOG['ee_omega'].append(ew)
    DATA_LOG['joint_pos'].append(jp); DATA_LOG['gripper_cmd'].append(gripper_val)

# =====================================================================
# 3. KINEMATICS
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
    for i, (a, alpha, d, th0) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(a, alpha, d, q[i] + th0)
    T = T @ T_TOOL
    return T[:3, 3].copy(), T[:3, :3].copy()

def _fk_frame3(q):
    T = np.eye(4)
    for i in range(3):
        a, alpha, d, th0 = DH_PARAMS[i]
        T = T @ dh_matrix(a, alpha, d, q[i] + th0)
    return T

def rpy_to_rot(roll, pitch, yaw):
    cr,sr = np.cos(roll), np.sin(roll)
    cp,sp = np.cos(pitch),np.sin(pitch)
    cy,sy = np.cos(yaw),  np.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx

def rot_to_rpy(R):
    pitch = math.asin(np.clip(R[0,2], -1, 1))
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(-R[1,2], R[2,2])
        yaw  = math.atan2(-R[0,1], R[0,0])
    else:
        roll = math.atan2(R[2,1], R[1,1])
        yaw  = 0.0
    return roll, pitch, yaw

def jacobian_pos(q, fd_eps=1e-7):
    Jv = np.zeros((3, 6))
    for i in range(6):
        qp, qm = q.copy(), q.copy()
        qp[i] += fd_eps; qm[i] -= fd_eps
        pp, _ = fk(qp); pm, _ = fk(qm)
        Jv[:, i] = (pp - pm) / (2 * fd_eps)
    return Jv

# =====================================================================
# KINEMATIC WEIGHTS
# =====================================================================
JOINT_PENALTY_WEIGHTS = np.diag([
    0.1,   # J1 (Base) - Heavy penalty
    0.1,   # J2 (Shoulder) - Heavy penalty
    0.1,   # J3 (Elbow) - Heavy penalty
    0.1,   # J4 (Wrist Roll) - LOWEST penalty (Prioritized over J6)
    0.5,   # J5 (Wrist Pitch) - Normal small joint penalty
    0.4    # J6 (Wrist Yaw) - Higher penalty than J4
])

def inverse_kinematics_pos(target_pos, initial_guess, max_iter=300, tol=1e-4, alpha=0.5, lambda_=0.01):
    """Weighted DLS for Position-Only tracking."""
    q = np.array(initial_guess, dtype=float)
    for _ in range(max_iter):
        pos_curr, _ = fk(q)
        error = target_pos - pos_curr
        if np.linalg.norm(error) < tol:
            break
        Jv = jacobian_pos(q)
        
        # Weighted DLS Formulation: A = (J^T * J + lambda^2 * W)
        A = Jv.T @ Jv + (lambda_**2) * JOINT_PENALTY_WEIGHTS
        
        # Solve for dq using the weighted matrix
        dq = np.linalg.solve(A, Jv.T @ error)
        q = q + alpha * dq
    return q

def _refine_full(q, t_pos, t_rot, iters=20, lam=0.05, alpha=0.5):
    """Weighted DLS polish for full 6-DOF tracking."""
    eps = 1e-7
    for _ in range(iters):
        pc, Rc = fk(q)
        ep  = t_pos - pc
        Re  = t_rot @ Rc.T
        eo  = 0.5 * np.array([Re[2,1]-Re[1,2], Re[0,2]-Re[2,0], Re[1,0]-Re[0,1]])
        err = np.concatenate([ep, eo])
        
        if np.linalg.norm(err) < 1e-5:
            break
            
        J = np.zeros((6, 6))
        for i in range(6):
            qp, qm = q.copy(), q.copy()
            qp[i] += eps; qm[i] -= eps
            pp, Rp = fk(qp); pm, Rm = fk(qm)
            J[:3, i] = (pp - pm) / (2*eps)
            dR = (Rp - Rm) / (2*eps)
            S  = dR @ Rc.T
            J[3,i]=S[2,1]; J[4,i]=S[0,2]; J[5,i]=S[1,0]
            
        # Apply the exact same Joint Penalty Weights here
        A = J.T @ J + (lam**2) * JOINT_PENALTY_WEIGHTS
        
        # Solve for dq
        q = q + alpha * np.linalg.solve(A, J.T @ err)
    return q

# =====================================================================
# 4. FULL 6-DOF IK  (decoupled: position step + wrist step + DLS polish)
# =====================================================================
def _ik_arm(p_wcp, q_prev):
    """Closed-form joints 1-3 for wrist centre p_wcp."""
    a1,a2,a3 = DH_PARAMS[0,0], DH_PARAMS[1,0], DH_PARAMS[2,0]
    d1,d4    = DH_PARAMS[0,2], DH_PARAMS[3,2]
    px,py,pz = p_wcp
    t1  = np.arctan2(py, px)
    r   = np.sqrt(px**2 + py**2) - a1
    z   = pz - d1
    L1  = a2
    L2  = np.sqrt(a3**2 + d4**2)
    D   = np.clip((r**2+z**2-L1**2-L2**2) / (2*L1*L2), -1.0, 1.0)
    phi = np.arctan2(a3, d4)
    best, best_err = None, 1e9
    for sign in (+1, -1):
        t3r = np.arctan2(sign * np.sqrt(1 - D**2), D)
        t3  = t3r - phi
        t2  = np.arctan2(z, r) - np.arctan2(L2*np.sin(t3r), L1 + L2*np.cos(t3r))
        sol = np.array([t1, t2, t3])
        err = np.linalg.norm(sol - q_prev[:3])
        if err < best_err:
            best_err, best = err, sol
    return best

def _ik_wrist(q123, R_target, q4_prev, q6_prev):
    """ZYZ wrist IK. Joint 4 has priority at singularity; joint 6 holds."""
    R03 = _fk_frame3(q123)[:3, :3]
    R36 = R03.T @ R_target
    t5  = np.arctan2(np.sqrt(R36[0,2]**2 + R36[1,2]**2), R36[2,2])
    if abs(np.sin(t5)) > 1e-6:
        t4 = np.arctan2( R36[1,2],  R36[0,2])
        t6 = np.arctan2( R36[2,1], -R36[2,0])
    else:
        t6 = q6_prev
        if np.cos(t5) > 0:
            t4 = np.arctan2(-R36[0,1], R36[0,0]) - t6
        else:
            t4 = np.arctan2( R36[0,1],-R36[0,0]) + t6
    return np.array([t4, t5, t6])


def solve_ik_full(t_pos, t_rot, q_prev):
    """
    Full 6-DOF IK: given target position + rotation matrix, returns joint angles.
    """
    tool_z = t_rot @ T_TOOL[:3, 2]
    p_wcp  = t_pos - WRIST_OFFSET * tool_z
    q123   = _ik_arm(p_wcp, q_prev)
    q456   = _ik_wrist(q123, t_rot, q_prev[3], q_prev[5])
    q      = _refine_full(np.concatenate([q123, q456]), t_pos, t_rot)
    return q

def dh_to_sim(q):
    return JOINT_SIGN * q

# =====================================================================
# 5. TRAJECTORY GENERATION
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
        configs.append((1.0 - s) * q_start + s * q_target)
    return configs

# =====================================================================
# 6. COPPELIA HELPERS
# =====================================================================
TARGET_ARM   = '/yaskawa'
STR_SIGNAL   = TARGET_ARM + '_executedMovId'
_exec_mov_id = 'notReady'

def wait_for_movement(sim, move_id, cup_handle, ee_handle, timeout=60.0):
    global _exec_mov_id
    _exec_mov_id = 'notReady'
    t0 = time.time()
    while _exec_mov_id != move_id:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Movement '{move_id}' timed out after {timeout} s")
        s = sim.getStringSignal(STR_SIGNAL)
        if s is not None:
            _exec_mov_id = s.decode() if isinstance(s, bytes) else s
        log_simulation_data(sim, cup_handle, ee_handle)
        time.sleep(0.05)

def dispatch(sim, configs_sim, times, move_id, gripper_vel=0.0):
    assert len(configs_sim) == len(times)
    g_list = [float(g) for g in gripper_vel] if isinstance(gripper_vel, list) \
             else [float(gripper_vel)] * len(times)
    md = {
        'id': move_id, 'type': 'pts', 'times': times,
        'j1': [float(q[0]) for q in configs_sim],
        'j2': [float(q[1]) for q in configs_sim],
        'j3': [float(q[2]) for q in configs_sim],
        'j4': [float(q[3]) for q in configs_sim],
        'j5': [float(q[4]) for q in configs_sim],
        'j6': [float(q[5]) for q in configs_sim],
        'gripper': g_list,
    }
    sim.callScriptFunction(f'remoteApi_movementDataFunction@{TARGET_ARM}', sim.scripttype_childscript, md)
    sim.callScriptFunction(f'remoteApi_executeMovement@{TARGET_ARM}',      sim.scripttype_childscript, move_id)

# =====================================================================
# 7. PLOTTING
# =====================================================================
def plot_results():
    print("\nGenerating Plots...")
    t = np.array(DATA_LOG['t'])
    if len(t) == 0:
        print("No data logged to plot.")
        return
    cup_pos   = np.array(DATA_LOG['cup_pos']);   cup_vel   = np.array(DATA_LOG['cup_vel'])
    cup_ori   = np.array(DATA_LOG['cup_ori']);   cup_omega = np.array(DATA_LOG['cup_omega'])
    ee_pos    = np.array(DATA_LOG['ee_pos']);    ee_vel    = np.array(DATA_LOG['ee_vel'])
    ee_ori    = np.array(DATA_LOG['ee_ori']);    ee_omega  = np.array(DATA_LOG['ee_omega'])

    fig1,(ax1,ax2) = plt.subplots(2,1,figsize=(10,8)); fig1.suptitle("1. Cup Trajectory")
    ax1.plot(t,cup_pos[:,0],label='X'); ax1.plot(t,cup_pos[:,1],label='Y'); ax1.plot(t,cup_pos[:,2],label='Z')
    ax1.set_ylabel('Position (m)'); ax1.legend(); ax1.grid(True)
    ax2.plot(t,cup_vel[:,0],label='Vx'); ax2.plot(t,cup_vel[:,1],label='Vy'); ax2.plot(t,cup_vel[:,2],label='Vz')
    ax2.set_ylabel('Velocity (m/s)'); ax2.set_xlabel('Time (s)'); ax2.legend(); ax2.grid(True)

    fig2,(ax3,ax4) = plt.subplots(2,1,figsize=(10,8)); fig2.suptitle("2. End-Effector Linear Trajectory")
    ax3.plot(t,ee_pos[:,0],label='X'); ax3.plot(t,ee_pos[:,1],label='Y'); ax3.plot(t,ee_pos[:,2],label='Z')
    ax3.set_ylabel('Position (m)'); ax3.legend(); ax3.grid(True)
    ax4.plot(t,ee_vel[:,0],label='Vx'); ax4.plot(t,ee_vel[:,1],label='Vy'); ax4.plot(t,ee_vel[:,2],label='Vz')
    ax4.set_ylabel('Velocity (m/s)'); ax4.set_xlabel('Time (s)'); ax4.legend(); ax4.grid(True)

    fig3 = plt.figure(figsize=(8,8)); ax5 = fig3.add_subplot(111,projection='3d')
    fig3.suptitle("2b. End-Effector 3D Path")
    ax5.plot(ee_pos[:,0],ee_pos[:,1],ee_pos[:,2],color='blue',linewidth=2); ax5.legend()

    fig4,(ax6,ax7) = plt.subplots(2,1,figsize=(10,8)); fig4.suptitle("3. EE Angular Trajectory")
    eu = np.unwrap(ee_ori, axis=0)
    ax6.plot(t,np.degrees(eu[:,0]),label='Roll'); ax6.plot(t,np.degrees(eu[:,1]),label='Pitch'); ax6.plot(t,np.degrees(eu[:,2]),label='Yaw')
    ax6.set_ylabel('Orientation (°)'); ax6.legend(); ax6.grid(True)
    ax7.plot(t,ee_omega[:,0],label='Wx'); ax7.plot(t,ee_omega[:,1],label='Wy'); ax7.plot(t,ee_omega[:,2],label='Wz')
    ax7.set_ylabel('Angular Vel (rad/s)'); ax7.set_xlabel('Time (s)'); ax7.legend(); ax7.grid(True)

    fig5,(ax8,ax9) = plt.subplots(2,1,figsize=(10,8)); fig5.suptitle("4. Cup Angular Trajectory")
    cu = np.unwrap(cup_ori, axis=0)
    ax8.plot(t,np.degrees(cu[:,0]),label='Roll'); ax8.plot(t,np.degrees(cu[:,1]),label='Pitch'); ax8.plot(t,np.degrees(cu[:,2]),label='Yaw')
    ax8.set_ylabel('Orientation (°)'); ax8.legend(); ax8.grid(True)
    ax9.plot(t,cup_omega[:,0],label='Wx'); ax9.plot(t,cup_omega[:,1],label='Wy'); ax9.plot(t,cup_omega[:,2],label='Wz')
    ax9.set_ylabel('Angular Vel (rad/s)'); ax9.set_xlabel('Time (s)'); ax9.legend(); ax9.grid(True)

    jp = np.array(DATA_LOG['joint_pos'])
    if jp.ndim == 2 and jp.shape[1] == 6:
        fig6, axes_j = plt.subplots(6,1,figsize=(12,14),sharex=True); fig6.suptitle("5. Joint Profiles")
        for idx,c in enumerate(['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']):
            axes_j[idx].plot(t,np.degrees(jp[:,idx]),color=c,linewidth=1.5,label=f'Joint {idx+1}')
            axes_j[idx].set_ylabel('Angle (°)',fontsize=8); axes_j[idx].legend(loc='upper right',fontsize=8); axes_j[idx].grid(True)
        axes_j[-1].set_xlabel('Time (s)'); fig6.tight_layout(rect=[0,0,1,0.97])

    gc = np.array(DATA_LOG['gripper_cmd'])
    fig7,ax_g = plt.subplots(figsize=(10,4)); fig7.suptitle("6. Gripper Command")
    ax_g.plot(t,gc,color='tab:blue',linewidth=2)
    ax_g.axhline(G_CLOSE,color='tab:red',   linewidth=1,linestyle=':',alpha=0.6,label=f'CLOSE ({G_CLOSE:+.3f})')
    ax_g.axhline(G_HOLD, color='tab:orange',linewidth=1,linestyle=':',alpha=0.6,label=f'HOLD  ({G_HOLD:+.3f})')
    ax_g.axhline(G_OPEN, color='tab:green', linewidth=1,linestyle=':',alpha=0.6,label=f'OPEN  ({G_OPEN:+.3f})')
    ax_g.set_ylim(G_CLOSE-0.01, G_OPEN+0.01); ax_g.legend(fontsize=10)
    ax_g.set_xlabel('Time (s)'); ax_g.set_ylabel('Gripper Velocity Command')
    ax_g.grid(True,axis='y',linestyle=':'); fig7.tight_layout(rect=[0,0,1,0.95])
    plt.show()

# =====================================================================
# 8. MAIN
# =====================================================================
if __name__ == "__main__":
    print("--- Connecting to CoppeliaSim ---")

    joint_handles = [sim.getObject(f'{TARGET_ARM}/joint{i+1}') for i in range(6)]

    try:    cup_handle = sim.getObject('./Cup')
    except: cup_handle = -1; print("Warning: './Cup' not found.")

    try:    ee_handle = sim.getObject('./gripperEF')
    except: ee_handle = -1; print("Warning: './gripperEF' not found.")

    N_STEPS = int(TARGET_DUR / DT)
    times   = [i * DT for i in range(N_STEPS)]

    q_sim_current = np.array([sim.getJointPosition(h) for h in joint_handles])
    q_dh_current  = JOINT_SIGN * q_sim_current

    # -----------------------------------------------------------------
    # Phase 3: Placing trajectories
    # -----------------------------------------------------------------
    print("\n--- Phase 3: Calculating Placing Trajectories ---")
    q_current_placing = q_dh_current.copy()
    all_placing_trajectories = []

    for i, target_pos in enumerate(PLACE_POSITIONS):
        print(f"  Placing Position {i+1}/{len(PLACE_POSITIONS)}...")
        q_target_placing  = inverse_kinematics_pos(target_pos, q_current_placing)
        placing_configs   = build_joint_trajectory(q_current_placing, q_target_placing, n_steps=IK_STEPS)
        placing_sim       = [dh_to_sim(q) for q in placing_configs]
        N_PL = int(PLACE_DUR / DT)
        all_placing_trajectories.append({
            'configs': resample_to_n(placing_sim, N_PL),
            'times':   [k * DT for k in range(N_PL)],
            'id':      f'waypoint_path_placing_{i+1}',
        })
        q_current_placing = q_target_placing

    # -----------------------------------------------------------------
    # Phase 4: Lift Trajectory
    # -----------------------------------------------------------------
    print("\n--- Phase 4: Calculating Lift Trajectory ---")
    target_pos_lift = np.array([0.64, 0.0, 0.70])
    print(f"  Target lift position: {target_pos_lift}...")
    q_target_lift = inverse_kinematics_pos(target_pos_lift, q_current_placing)
    lift_configs_dh = build_joint_trajectory(q_current_placing, q_target_lift, n_steps=IK_STEPS)
    lift_sim = [dh_to_sim(q) for q in lift_configs_dh]
    
    N_LIFT = int(LIFT_DUR / DT)
    lift_trajectory = {
        'configs': resample_to_n(lift_sim, N_LIFT),
        'times':   [k * DT for k in range(N_LIFT)],
        'id':      'waypoint_path_lift'
    }

    # -----------------------------------------------------------------
    # Phase 5: Execute
    # -----------------------------------------------------------------
    print("\n--- Starting Simulation ---")
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_advancing_running:
        time.sleep(0.1)
    print("Simulation running.")

    _dynamic_gripper_list = None
    _dynamic_gripper_t0   = None
    _current_gripper_cmd  = G_CLOSE

    # Placing
    print(f"\n--- Executing {len(all_placing_trajectories)} Placing Trajectories ---")
    for i, traj in enumerate(all_placing_trajectories):
        print(f"  Sending Placing {i+1} (Duration: {PLACE_DUR}s)...")
        dispatch(sim, traj['configs'], traj['times'], traj['id'], gripper_vel=G_HOLD)
        wait_for_movement(sim, traj['id'], cup_handle, ee_handle)
        print("  Done.")
        time.sleep(0.5)

    # -----------------------------------------------------------------
    # Close the gripper
    # -----------------------------------------------------------------
    print("\n--- Closing Gripper ---")
    _current_gripper_cmd = G_CLOSE
    N_CLOSE = int(1.0 / DT)  # Holds position for 1.0 second while closing
    
    final_q = all_placing_trajectories[-1]['configs'][-1]
    close_configs = [final_q] * N_CLOSE
    close_times   = [k * DT for k in range(N_CLOSE)]
    
    dispatch(sim, close_configs, close_times, 'waypoint_path_close', gripper_vel=G_CLOSE)
    wait_for_movement(sim, 'waypoint_path_close', cup_handle, ee_handle)
    print("  Gripper closed.")
    time.sleep(0.5)

    # -----------------------------------------------------------------
    # NEW: Lift to final position
    # -----------------------------------------------------------------
    print("\n--- Lifting to Final Position ---")
    # Execute the lift trajectory while maintaining the G_CLOSE velocity command
    dispatch(sim, lift_trajectory['configs'], lift_trajectory['times'], lift_trajectory['id'], gripper_vel=G_CLOSE)
    wait_for_movement(sim, lift_trajectory['id'], cup_handle, ee_handle)
    print("  Lift complete.")
    time.sleep(0.5)

    if input("\nPress Enter to stop the simulation...") is not None:
        sim.stopSimulation()
        print("\nSimulation stopped.")

    # plot_results()