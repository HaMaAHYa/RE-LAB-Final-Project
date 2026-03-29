"""
Yaskawa GP8 Pick-and-Place Script for CoppeliaSim
-------------------------------------------------
This script calculates the joint angles required to move a robotic arm
through a sequence of Cartesian (XYZ) waypoints using Inverse Kinematics (IK),
generates smooth trajectories, and sends them to CoppeliaSim.
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
import numpy as np
import math
import time

# =====================================================================
# 1. USER CONFIGURATION & SETUP
# =====================================================================
# Move all settings up here so they are easy to find and change.
client = RemoteAPIClient()
sim    = client.getObject('sim')
Pi = math.pi

# Timing and Simulation Settings
DT = 0.05               # Must match CoppeliaSim's simulation timestep
IK_STEPS = 40           # Number of waypoints to generate for smooth paths
G_CLOSE = -0.04       # Gripper state: +0.04 = open, -0.04 = close, 0.0 = hold
G_OPEN = 0.04        # Gripper state: +0.04 = open, -0.04 = close,
G_HOLD = 0.0         # Gripper state: +0.04 = open, -0.04 = close, 0.0 = hold
PLACE_DUR = 3.0         # How long each placing motion should take (seconds)

# Yaskawa GP8 Classical DH Parameters [a, alpha, d, theta_offset]
# These numbers define the physical lengths and twists of the robot's links.
DH_PARAMS = np.array([
    [0.040,  -Pi/2,   0.330,   0.0  ],
    [0.345,   0.0,    0.0,    -Pi/2 ],
    [0.040,  -Pi/2,   0.0,     0.0  ],
    [0.0,     Pi/2,   0.340,   0.0  ],
    [0.0,    -Pi/2,   0.0,     0.0  ],
    [0.0,     0.0,    0.24133, 0.0  ],
], dtype=float)

# Tool Center Point (TCP) Transform
# This shifts the calculation from the robot's wrist exactly to the tip of the gripper.
T_TOOL = np.array([
    [1, 0, 0,  0.00007],
    [0, 1, 0, -0.00023],
    [0, 0, 1,  0.01867],
    [0, 0, 0,  1.0    ],
], dtype=float)

# Joint Sign Map: +1 = DH model matches simulation, -1 = rotation is flipped in sim
JOINT_SIGN = np.array([+1, +1, -1, +1, -1, +1], dtype=float)

# ---------------- Waypoints ----------------
cup = sim.getObject('./20cmHighWallL[1]/Cup/cup_pose')
cup_pos = sim.getObjectPosition(cup, sim.handle_world)
print(cup_pos)
PLACE_POSITIONS = [
    np.array([cup_pos[0] - 0.19, cup_pos[1], cup_pos[2]]),   
    np.array(cup_pos),   
]

# Time for Phase 1 (Approach) + Time for Phase 2 (Grabbing)
APPROACH_DUR = IK_STEPS * DT

TARGET_DUR = APPROACH_DUR


# =====================================================================
# 2. DATA LOGGING & MATH
# =====================================================================

# Global storage for graphs
DATA_LOG = {
    't': [],
    'cup_pos': [], 'cup_vel': [],
    'cup_ori': [], 'cup_omega': [],
    'ee_pos': [], 'ee_vel': [],
    'ee_ori': [], 'ee_omega': [],
    'joint_pos': [],    # All 6 joint angles (sim frame, radians)
    'gripper_cmd': [],  # Actual gripper velocity command over time
}

# Gripper command tracking globals
_current_gripper_cmd  = G_HOLD   # scalar fallback used for constant phases
_dynamic_gripper_list = None     # set to the dynamic list during the grabbing phase
_dynamic_gripper_t0   = None     # sim time when the grabbing dispatch was sent

def log_simulation_data(sim, cup_handle, ee_handle):
    """Samples the physical state of objects directly from CoppeliaSim."""
    global _current_gripper_cmd, _dynamic_gripper_list, _dynamic_gripper_t0
    t = sim.getSimulationTime()
    
    # Avoid duplicate polling in the same simulation step
    if len(DATA_LOG['t']) > 0 and t <= DATA_LOG['t'][-1]:
        return

    # Grab Cup Data
    if cup_handle != -1:
        cp = sim.getObjectPosition(cup_handle, sim.handle_world)
        cv, cw = sim.getObjectVelocity(cup_handle)
        co = sim.getObjectOrientation(cup_handle, sim.handle_world)
    else:
        cp, cv, cw, co = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]

    # Grab End-Effector Data
    if ee_handle != -1:
        ep = sim.getObjectPosition(ee_handle, sim.handle_world)
        ev, ew = sim.getObjectVelocity(ee_handle)
        eo = sim.getObjectOrientation(ee_handle, sim.handle_world)
    else:
        ep, ev, ew, eo = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]

    # Log all 6 joint positions (sim frame)
    if 'joint_handles' in globals() and joint_handles:
        jp = [sim.getJointPosition(h) for h in joint_handles]
    else:
        jp = [0.0] * 6

    # Resolve current gripper command: interpolate from dynamic list if in grabbing phase,
    # otherwise use the scalar set for the current phase.
    if _dynamic_gripper_list is not None and _dynamic_gripper_t0 is not None:
        elapsed = t - _dynamic_gripper_t0
        idx = int(elapsed / DT)
        idx = max(0, min(idx, len(_dynamic_gripper_list) - 1))
        gripper_val = _dynamic_gripper_list[idx]
    else:
        gripper_val = _current_gripper_cmd

    DATA_LOG['t'].append(t)
    DATA_LOG['cup_pos'].append(cp)
    DATA_LOG['cup_vel'].append(cv)
    DATA_LOG['cup_ori'].append(co)
    DATA_LOG['cup_omega'].append(cw)
    DATA_LOG['ee_pos'].append(ep)
    DATA_LOG['ee_vel'].append(ev)
    DATA_LOG['ee_ori'].append(eo)
    DATA_LOG['ee_omega'].append(ew)
    DATA_LOG['joint_pos'].append(jp)
    DATA_LOG['gripper_cmd'].append(gripper_val)

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

def jacobian_pos(q, fd_eps=1e-7):
    Jv = np.zeros((3, 6))
    for i in range(6):
        q_p, q_m = q.copy(), q.copy()
        q_p[i] += fd_eps  
        q_m[i] -= fd_eps  
        p_p, _ = fk(q_p)
        p_m, _ = fk(q_m)
        Jv[:, i] = (p_p - p_m) / (2 * fd_eps)
    return Jv

def inverse_kinematics_pos(target_pos, initial_guess, max_iter=300, tol=1e-4, alpha=0.5, lambda_=0.01):
    q = np.array(initial_guess, dtype=float)
    for _ in range(max_iter):
        pos_curr, _ = fk(q)
        error = target_pos - pos_curr          
        if np.linalg.norm(error) < tol:        
            break
        Jv    = jacobian_pos(q)                
        A     = Jv @ Jv.T + lambda_**2 * np.eye(3) 
        J_dls = Jv.T @ np.linalg.inv(A)       
        q     = q + alpha * (J_dls @ error)
    return q

def dh_to_sim(q):
    return JOINT_SIGN * q


# =====================================================================
# 3. TRAJECTORY GENERATION (PATH PLANNING)
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
        q = (1.0 - s) * q_start + s * q_target
        configs.append(q)
    return configs


# =====================================================================
# 4. COPPELIA SIMULATION HELPERS
# =====================================================================

TARGET_ARM   = '/yaskawa'
STR_SIGNAL   = TARGET_ARM + '_executedMovId'
_exec_mov_id = 'notReady'

def wait_for_movement(sim, move_id, cup_handle, ee_handle, timeout=60.0):
    """Pauses Python until the movement is done, polling logs in the background."""
    global _exec_mov_id
    _exec_mov_id = 'notReady'
    t0 = time.time()
    while _exec_mov_id != move_id:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Movement '{move_id}' timed out after {timeout} s")
        s = sim.getStringSignal(STR_SIGNAL)
        if s is not None:
            _exec_mov_id = s.decode() if isinstance(s, bytes) else s
            
        # Perform continuous logging while we wait!
        log_simulation_data(sim, cup_handle, ee_handle)
        
        time.sleep(0.05)

def dispatch(sim, configs_sim, times, move_id, gripper_vel=0.0):
    """
    MODIFIED: Now accepts a list for gripper_vel, allowing the gripper 
    state to change smoothly mid-movement.
    """
    assert len(configs_sim) == len(times), "configs_sim length != times length"
    
    # If gripper_vel is a single number, stretch it to cover all steps.
    # If it is already a list, use it directly.
    if isinstance(gripper_vel, list):
        g_list = [float(g) for g in gripper_vel]
    else:
        g_list = [float(gripper_vel)] * len(times)
    
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
        'gripper': g_list,
    }
    sim.callScriptFunction(f'remoteApi_movementDataFunction@{TARGET_ARM}', sim.scripttype_childscript, md)
    sim.callScriptFunction(f'remoteApi_executeMovement@{TARGET_ARM}', sim.scripttype_childscript, move_id)


# =====================================================================
# 6. PLOTTING FUNCTION
# =====================================================================
def plot_results():
    print("\nGenerating Plots...")
    t = np.array(DATA_LOG['t'])
    if len(t) == 0:
        print("No data logged to plot.")
        return

    cup_pos = np.array(DATA_LOG['cup_pos'])
    cup_vel = np.array(DATA_LOG['cup_vel'])
    cup_ori = np.array(DATA_LOG['cup_ori'])
    cup_omega = np.array(DATA_LOG['cup_omega'])
    
    ee_pos = np.array(DATA_LOG['ee_pos'])
    ee_vel = np.array(DATA_LOG['ee_vel'])
    ee_ori = np.array(DATA_LOG['ee_ori'])
    ee_omega = np.array(DATA_LOG['ee_omega'])

    # -- 1. Cup Position and Velocity --
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig1.suptitle("1. Cup Trajectory over Time")
    ax1.plot(t, cup_pos[:,0], label='X')
    ax1.plot(t, cup_pos[:,1], label='Y')
    ax1.plot(t, cup_pos[:,2], label='Z')
    ax1.set_ylabel('Position (m)')
    ax1.legend(); ax1.grid(True)

    ax2.plot(t, cup_vel[:,0], label='Vx')
    ax2.plot(t, cup_vel[:,1], label='Vy')
    ax2.plot(t, cup_vel[:,2], label='Vz')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_xlabel('Time (s)')
    ax2.legend(); ax2.grid(True)

    # -- 2. End-Effector Linear Trajectory --
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
    fig2.suptitle("2. End-Effector Linear Trajectory")
    ax3.plot(t, ee_pos[:,0], label='X')
    ax3.plot(t, ee_pos[:,1], label='Y')
    ax3.plot(t, ee_pos[:,2], label='Z')
    ax3.set_ylabel('Position (m)')
    ax3.legend(); ax3.grid(True)

    ax4.plot(t, ee_vel[:,0], label='Vx')
    ax4.plot(t, ee_vel[:,1], label='Vy')
    ax4.plot(t, ee_vel[:,2], label='Vz')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_xlabel('Time (s)')
    ax4.legend(); ax4.grid(True)

    # -- 2b. End-Effector 3D Path --
    fig3 = plt.figure(figsize=(8, 8))
    ax5 = fig3.add_subplot(111, projection='3d')
    fig3.suptitle("2b. End-Effector 3D Path")
    ax5.plot(ee_pos[:,0], ee_pos[:,1], ee_pos[:,2], label='TCP Path', color='blue', linewidth=2)
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_zlabel('Z (m)')
    ax5.legend()

    # -- 3. End-Effector Angular Trajectory --
    fig4, (ax6, ax7) = plt.subplots(2, 1, figsize=(10, 8))
    fig4.suptitle("3. End-Effector Angular Trajectory (Orientation & Omega)")
    
    # Unwrap angles to prevent jumpy charts from -180 to +180 flips
    ee_ori_unwrapped = np.unwrap(ee_ori, axis=0) 

    ax6.plot(t, np.degrees(ee_ori_unwrapped[:,0]), label='Alpha (Roll)')
    ax6.plot(t, np.degrees(ee_ori_unwrapped[:,1]), label='Beta (Pitch)')
    ax6.plot(t, np.degrees(ee_ori_unwrapped[:,2]), label='Gamma (Yaw)')
    ax6.set_ylabel('Orientation (Degrees)')
    ax6.legend(); ax6.grid(True)

    ax7.plot(t, ee_omega[:,0], label='Omega X')
    ax7.plot(t, ee_omega[:,1], label='Omega Y')
    ax7.plot(t, ee_omega[:,2], label='Omega Z')
    ax7.set_ylabel('Angular Vel (rad/s)')
    ax7.set_xlabel('Time (s)')
    ax7.legend(); ax7.grid(True)
    
    # -- 4. Cup Angular Trajectory --
    fig5, (ax8, ax9) = plt.subplots(2, 1, figsize=(10, 8))
    fig5.suptitle("4. Cup Angular Trajectory (Orientation & Omega)")
    
    cup_ori_unwrapped = np.unwrap(cup_ori, axis=0) 

    ax8.plot(t, np.degrees(cup_ori_unwrapped[:,0]), label='Alpha (Roll)')
    ax8.plot(t, np.degrees(cup_ori_unwrapped[:,1]), label='Beta (Pitch)')
    ax8.plot(t, np.degrees(cup_ori_unwrapped[:,2]), label='Gamma (Yaw)')
    ax8.set_ylabel('Orientation (Degrees)')
    ax8.legend(); ax8.grid(True)

    ax9.plot(t, cup_omega[:,0], label='Omega X')
    ax9.plot(t, cup_omega[:,1], label='Omega Y')
    ax9.plot(t, cup_omega[:,2], label='Omega Z')
    ax9.set_ylabel('Angular Vel (rad/s)')
    ax9.set_xlabel('Time (s)')
    ax9.legend(); ax9.grid(True)

    # -- 5. All 6 Joint Profiles --
    joint_pos = np.array(DATA_LOG['joint_pos'])
    if joint_pos.ndim == 2 and joint_pos.shape[1] == 6:
        fig6, axes_j = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
        fig6.suptitle("5. Joint Angle Profiles vs Time", fontsize=13)
        colors_j = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        for idx in range(6):
            axes_j[idx].plot(t, np.degrees(joint_pos[:, idx]),
                             color=colors_j[idx], linewidth=1.5,
                             label=f'Joint {idx+1}')
            axes_j[idx].set_ylabel('Angle (°)', fontsize=8)
            axes_j[idx].legend(loc='upper right', fontsize=8)
            axes_j[idx].grid(True)
        axes_j[-1].set_xlabel('Time (s)')
        fig6.tight_layout(rect=[0, 0, 1, 0.97])

    # -- 6. Gripper Command vs Time --
    gripper_cmd = np.array(DATA_LOG['gripper_cmd'])
    fig7, ax_g = plt.subplots(figsize=(10, 4))
    fig7.suptitle("6. Gripper Command vs Time", fontsize=13)

    ax_g.plot(t, gripper_cmd, color='tab:blue', linewidth=2)
    ax_g.set_xlabel('Time (s)')
    ax_g.set_ylabel('Gripper Velocity Command')

    # Draw reference lines so the named states are easy to read
    ax_g.axhline(G_CLOSE, color='tab:red',    linewidth=1, linestyle=':', alpha=0.6, label=f'G_CLOSE ({G_CLOSE:+.3f})')
    ax_g.axhline(G_HOLD,  color='tab:orange', linewidth=1, linestyle=':', alpha=0.6, label=f'G_HOLD  ({G_HOLD:+.3f})')
    ax_g.axhline(G_OPEN,  color='tab:green',  linewidth=1, linestyle=':', alpha=0.6, label=f'G_OPEN  ({G_OPEN:+.3f})')

    ax_g.set_ylim(G_CLOSE - 0.01, G_OPEN + 0.01)
    ax_g.legend(fontsize=10)
    ax_g.grid(True, axis='y', linestyle=':')
    fig7.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

# =====================================================================
# 5. MAIN EXECUTION SCRIPT
# =====================================================================

if __name__ == "__main__":
    print("--- Connecting to CoppeliaSim ---")

    joint_handles = [sim.getObject(f'{TARGET_ARM}/joint{i+1}') for i in range(6)]

    # Fetch Handles for Logging
    try:
        cup_handle = sim.getObject('./Cup')
    except:
        cup_handle = -1
        print("Warning: Object './Cup' not found. Cup data will be zeroed.")
        
    try:
        # Defaulting to gripperEF; if named differently, change it here
        ee_handle = sim.getObject('./gripperEF')
    except:
        ee_handle = -1
        print("Warning: Object './gripperEF' not found. EE data will be zeroed.")


    N_STEPS = int(TARGET_DUR / DT)
    times   = [i * DT for i in range(N_STEPS)]

    q_sim_current = np.array([sim.getJointPosition(h) for h in joint_handles])
    q_dh_current  = JOINT_SIGN * q_sim_current
    pos_start, _ = fk(q_dh_current)

    # -----------------------------------------------------------------
    # Phase 1: Smooth approach to the start of the grab
    # -----------------------------------------------------------------
    ik_configs_dh = [q_dh_current.copy()]
    q_seed = q_dh_current.copy()

    # -----------------------------------------------------------------
    # GRIPPER LOGIC: Calculate when to close the gripper
    # -----------------------------------------------------------------
    # We want it open/hold (0.0) until we finish the approach (reaching TARGETS[0]),
    # and closed (-0.04) while sweeping through the rest of the TARGETS array.

    # -----------------------------------------------------------------
    # Phase 3: Calculate Joint Space Trajectories for all Placing points
    # -----------------------------------------------------------------
    print("\n--- Phase 3: Calculating Joint Space Trajectories for Placing ---")
    q_current_placing = ik_configs_dh[-1]  
    all_placing_trajectories = []

    for i, target_pos in enumerate(PLACE_POSITIONS):
        print(f"  Calculating path for Placing Position {i+1}/{len(PLACE_POSITIONS)}...")
        q_target_placing = inverse_kinematics_pos(target_pos, q_current_placing)
        
        placing_configs_dh = build_joint_trajectory(q_current_placing, q_target_placing, n_steps=IK_STEPS)
        placing_configs_sim = [dh_to_sim(q) for q in placing_configs_dh]
        
        N_STEPS_PLACE = int(PLACE_DUR / DT)
        times_place   = [k * DT for k in range(N_STEPS_PLACE)]
        configs_sim_place = resample_to_n(placing_configs_sim, N_STEPS_PLACE)
        
        all_placing_trajectories.append({
            'configs': configs_sim_place,
            'times': times_place,
            'id': f'waypoint_path_placing_{i+1}'
        })
        q_current_placing = q_target_placing

    # -----------------------------------------------------------------
    # Phase 4: Execute Simulation
    # -----------------------------------------------------------------
    print("\n--- Starting Simulation ---")
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_advancing_running:
        time.sleep(0.1)
    print("Simulation running.") 
    
    # Grabbing done — switch back to scalar tracking
    _dynamic_gripper_list = None
    _dynamic_gripper_t0   = None
    _current_gripper_cmd  = G_CLOSE

    # Execute Placing (Keep gripper closed = G_CLOSE = -0.04)
    print(f"\n--- Executing {len(all_placing_trajectories)} Placing Trajectories ---")
    for i, traj in enumerate(all_placing_trajectories):
        move_id = traj['id']
        print(f"Sending Placing trajectory {i+1} to CoppeliaSim (Duration: {PLACE_DUR}s)...")
        
        dispatch(sim, traj['configs'], traj['times'], move_id, gripper_vel=G_HOLD)

        print(f"  Moving to Placing Target {i+1}...", end='', flush=True)
        wait_for_movement(sim, move_id, cup_handle, ee_handle)
        print("\r  Done.                         ")
        time.sleep(0.5) 
    
    
    # -----------------------------------------------------------------
    # Phase 5: REAL-TIME PRECISION REACH (No Freeze)
    # -----------------------------------------------------------------
    print("\n--- Phase 5: Real-Time Precision Reach ---")

    # 1. Capture current state
    q_sim_now = np.array([sim.getJointPosition(h) for h in joint_handles])
    q_dh_now = JOINT_SIGN * q_sim_now
    TCP_start, _ = fk(q_dh_now)
    
    cup_handle = 395 
    CC = np.array(sim.getObjectPosition(cup_handle, sim.handle_world))

    # 2. Simplified Trajectory (Fewer iterations to prevent freezing)
    configs_dh = []
    q_step = q_dh_now.copy()
    
    # Reduced steps and iterations for speed
    FAST_IK_STEPS = 20 
    for k in range(FAST_IK_STEPS):
        t = k / (FAST_IK_STEPS - 1)
        s = 6*t**5 - 15*t**4 + 10*t**3
        interp_pos = (1.0 - s) * TCP_start + s * CC
        
        # Lower max_iter to 100 to stop the 'freeze'
        q_step = inverse_kinematics_pos(interp_pos, q_step, max_iter=100, tol=1e-5, alpha=0.3)
        configs_dh.append(q_step.copy())

    # 3. Seamless Sync & Dispatch
    configs_sim = [dh_to_sim(q) for q in configs_dh]
    configs_sim[0] = q_sim_now 

    CONTACT_DUR = 2.0 
    times_contact = [k * DT for k in range(int(CONTACT_DUR / DT))]
    final_configs = resample_to_n(configs_sim, len(times_contact))
    
    print(f"Moving to CC: {CC}...")
    dispatch(sim, final_configs, times_contact, 'rt_reach', gripper_vel=G_CLOSE)
    wait_for_movement(sim, 'rt_reach', cup_handle, ee_handle)
    
    # 4. Final Diagnostic
    TCP_final = np.array(sim.getObjectPosition(ee_handle, sim.handle_world))
    final_dist = np.linalg.norm(TCP_final - CC)
    
    print(f"FINAL ERROR: {final_dist:.6f}m")

    # 5. JOINT LIMIT CHECKER
    print("\n--- Diagnostic: Joint State vs Limits ---")
    for i, h in enumerate(joint_handles):
        pos = np.degrees(sim.getJointPosition(h))
        # Check if any joint is near typical limits (usually -180, 180 or similar)
        print(f" Joint {i+1}: {pos:.2f}°")

    
    if (input("\nPress Enter to stop the simulation...") is not None):
        sim.stopSimulation()
        print("\nSimulation stopped.")
        
    # Generate the requested graphs
