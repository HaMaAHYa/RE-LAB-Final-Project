"""
Sequence of Operations:
- Passively scans for markers 1, 2, 3, 4 during the main pick-and-place operation.
- Saves the last known center coordinate.
- Uses the saved center to visually servo ArUco 0 at the end of the sequence.
- Flips Joint 4 (Wrist Roll).
- Rotates Joint 5 (Wrist Pitch).
- Moves the arm straight down to Z = +0.540 using a Cartesian linear trajectory.
- Fully opens gripper and holds the exact live simulation position.
- Retreats back on X, lifts up on Z, and returns Home via Cartesian space.
"""

import os
import cv2
import numpy as np
import time
import math
import threading
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# --- LINUX GUI FIX ---
# Sometimes OpenCV's `cv2.imshow` crashes on Linux machines due to Qt backend issues.
# These environment variables force OpenCV to use the 'xcb' backend and suppress debug warnings.
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

# =====================================================================
# 1. SHARED CONFIGURATION & SETUP
# =====================================================================

# ---------------------------------------------------------------------
# ArUco Markers
# ArUco markers are square fiducial markers (like simplified QR codes) designed 
# specifically for fast, robust computer vision. Every marker has a unique ID.
# Here, we assign logical "meanings" to specific IDs so the robot knows what they are.
# ---------------------------------------------------------------------
# ArUco Config
ARUCO_MEANINGS = {
    0:  "Home Position / Origin",
    1:  "Pick-up Zone A",
    2:  "Pick-up Zone B",
    3:  "Drop-off Zone",
    4:  "Obstacle Warning",
}

# Robot Config
Pi = math.pi
DT = 0.05               # Delta Time: CoppeliaSim step size (50ms).
IK_STEPS = 40           # Waypoint density for generated paths.
G_CLOSE = -0.04         # Gripper motor closing velocity.
G_OPEN  =  0.04         # Gripper motor opening velocity.
G_HOLD  =  0.0          # Gripper motor lock/stall state.
PLACE_DUR = 3.0         # Duration of the placing movement.
LIFT_DUR  = 3.0         # Duration of the lifting movement.

# ---------------------------------------------------------------------
# Denavit-Hartenberg (DH) & Tool Center Point (TCP)
# DH_PARAMS define the physical skeleton of the Yaskawa GP8 robot (link lengths, twists).
# T_TOOL pushes the mathematical calculation point from the robot's wrist up to the 
# actual tips of the gripper fingers.
# ---------------------------------------------------------------------
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

# JOINT_SIGN corrects mismatches between the math model and 3D simulation axes.
JOINT_SIGN = np.array([+1, +1, -1, +1, -1, +1], dtype=float)
WRIST_OFFSET = T_TOOL[2, 3]

APPROACH_DUR = IK_STEPS * DT
TARGET_DUR   = APPROACH_DUR

# Data Logging Globals to track simulation state over time.
DATA_LOG = {
    't': [],
    'cup_pos': [], 'cup_vel': [], 'cup_ori': [], 'cup_omega': [],
    'ee_pos':  [], 'ee_vel':  [], 'ee_ori':  [], 'ee_omega':  [],
    'joint_pos': [], 'gripper_cmd': [],
}
_current_gripper_cmd  = G_HOLD
_dynamic_gripper_list = None
_dynamic_gripper_t0   = None

# ---------------------------------------------------------------------
# Thread Locks (Mutexes)
# Because the Main Thread and Vision Thread run simultaneously, they might try to 
# read and write to `vision_state` at the exact same microsecond, causing a crash.
# `vision_state_lock` is a Mutex (Mutual Exclusion). It acts like a talking stick; 
# a thread must hold the lock to touch the dictionary, ensuring safe data sharing.
# ---------------------------------------------------------------------
# --- THREAD SHARED STATE FOR VISION ---
vision_state_lock = threading.Lock()
vision_state = {
    'target_center': None, # (x, y) pixels of the center of 1,2,3,4
    'id0_pos': None        # (x, y) pixels of ArUco 0
}


# =====================================================================
# 2. ARUCO VISION THREAD FUNCTIONS
# =====================================================================

def get_handle(sim, possible_names, fallback_id=None):
    """Safely attempts to find simulation objects by name, falling back if they don't exist."""
    for name in possible_names:
        try:
            handle = sim.getObject(name)
            if handle != -1: return handle
        except:
            continue
    return fallback_id

def get_image(sim, sensor_handle):
    """
    Pulls raw image buffer data directly from the CoppeliaSim virtual camera.
    Converts it from a flat 1D byte array into a 3D NumPy array (Height, Width, RGB),
    then flips it vertically and converts RGB to BGR (which OpenCV requires).
    """
    try:
        img, resolution = sim.getVisionSensorImg(sensor_handle)
        if not img or resolution[0] == 0: return None
        arr = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
        return cv2.flip(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), 0)
    except: 
        return None

def detect_and_annotate(frame, detector):
    """
    ArUco Detection
    1. Converts image to Grayscale (color isn't needed for ArUco and slows down processing).
    2. detector.detectMarkers scans for sharp square contours and decodes their internal grids.
    3. We average the 4 corners of the marker to find its exact center pixel (cx, cy).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    found_markers = {} 
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            mid = int(marker_id)
            c = corners[i][0]
            cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
            found_markers[mid] = (cx, cy)
            
            # Draw text labels on the video feed
            meaning = ARUCO_MEANINGS.get(mid, f"ID:{mid}")
            cv2.putText(frame, f"{meaning} ({cx},{cy})", (cx - 40, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, found_markers

def vision_thread_worker(stop_event):
    """
    BACKGROUND THREAD PROCESS:
    This runs in an infinite loop (until `stop_event` is triggered at script end).
    It constantly grabs frames, analyzes them, and updates the shared `vision_state` variable.
    """
    print("[Vision Thread] Connecting dedicated client...")
    client_vis = RemoteAPIClient()
    sim_vis = client_vis.getObject('sim')
    
    sensor_handle = get_handle(sim_vis, ['./Vision_sensor', 'Vision_sensor'], fallback_id=384)
    if sensor_handle is None or sensor_handle == -1:
        print("[Vision Thread] ✘ Error: Could not find Vision Sensor. Thread stopping.")
        return

    # Load the specific dictionary of ArUco tags we are using (4x4 pixel grid, up to 50 IDs)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    WINDOW_NAME = "RE-LAB ArUco Tracking"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("[Vision Thread] Started Tracking Window.")
    
    while not stop_event.is_set():
        frame = get_image(sim_vis, sensor_handle)
        if frame is not None:
            annotated, marker_centers = detect_and_annotate(frame, detector)
            
            # Calculate the mathematical center point between markers 1, 2, 3, and 4
            tgt_center = None
            valid_pts = [marker_centers[m] for m in [1, 2, 3, 4] if m in marker_centers]
            if len(valid_pts) == 4:
                avg_x = sum(p[0] for p in valid_pts) // 4
                avg_y = sum(p[1] for p in valid_pts) // 4
                tgt_center = (avg_x, avg_y)
                
            # --- CRITICAL SECTION: Using the Thread Lock ---
            # We safely update the global variables so the Main Thread can read them later.
            with vision_state_lock:
                if tgt_center is not None:
                    vision_state['target_center'] = tgt_center
                vision_state['id0_pos'] = marker_centers.get(0, None)

            # Safely read back the target to draw it on the screen
            with vision_state_lock:
                draw_center = vision_state['target_center']
            
            if draw_center is not None:
                cv2.circle(annotated, draw_center, 5, (0, 0, 255), -1)
                cv2.putText(annotated, "SAVED TARGET", (draw_center[0] + 10, draw_center[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow(WINDOW_NAME, annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Vision Thread] Manual quit requested.")
            break
            
    cv2.destroyAllWindows()
    print("[Vision Thread] Closed gracefully.")


# =====================================================================
# 3. ROBOT KINEMATICS & TRAJECTORY FUNCTIONS (MAIN THREAD)
# =====================================================================

def log_simulation_data(sim, cup_handle, ee_handle, joint_handles):
    """Logs simulation metrics (pos, vel, orientation) for plotting later."""
    global _current_gripper_cmd, _dynamic_gripper_list, _dynamic_gripper_t0
    t = sim.getSimulationTime()
    if len(DATA_LOG['t']) > 0 and t <= DATA_LOG['t'][-1]: return
    
    cp, cv, cw, co = [0,0,0],[0,0,0],[0,0,0],[0,0,0]
    if cup_handle != -1:
        cp = sim.getObjectPosition(cup_handle, sim.handle_world)
        cv, cw = sim.getObjectVelocity(cup_handle)
        co = sim.getObjectOrientation(cup_handle, sim.handle_world)
        
    ep, ev, ew, eo = [0,0,0],[0,0,0],[0,0,0],[0,0,0]
    if ee_handle != -1:
        ep = sim.getObjectPosition(ee_handle, sim.handle_world)
        ev, ew = sim.getObjectVelocity(ee_handle)
        eo = sim.getObjectOrientation(ee_handle, sim.handle_world)
        
    jp = [sim.getJointPosition(h) for h in joint_handles] if joint_handles else [0.0] * 6
    
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

def dh_matrix(a, alpha, d, theta):
    """Calculates the 4x4 transformation matrix for a single robot link."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1.0 ],
    ])

def fk(q):
    """Calculates the End-Effector XYZ position based on 6 joint angles."""
    T = np.eye(4)
    for i, (a, alpha, d, th0) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(a, alpha, d, q[i] + th0)
    T = T @ T_TOOL
    return T[:3, 3].copy(), T[:3, :3].copy()

def jacobian_pos(q, fd_eps=1e-7):
    """Numerically calculates the Jacobian matrix to map joint velocities to task-space velocities."""
    Jv = np.zeros((3, 6))
    for i in range(6):
        qp, qm = q.copy(), q.copy()
        qp[i] += fd_eps; qm[i] -= fd_eps
        pp, _ = fk(qp); pm, _ = fk(qm)
        Jv[:, i] = (pp - pm) / (2 * fd_eps)
    return Jv

# ---------------------------------------------------------------------
# Joint Penalty Weights for Inverse Kinematics
# Because a 6-axis robot has redundant ways to reach an XYZ coordinate, the math
# can sometimes cause the robot to twist violently (singularities). 
# By adding weights to the Damped Least Squares matrix, we "penalize" the solver 
# for trying to move joints 5 and 6 (the wrist), forcing it to solve the reach 
# using the base and shoulder joints (1-4) primarily.
# ---------------------------------------------------------------------
JOINT_PENALTY_WEIGHTS = np.diag([0.1, 0.1, 0.1, 0.1, 0.5, 0.4])

def inverse_kinematics_pos(target_pos, initial_guess, max_iter=300, tol=1e-4, alpha=0.5, lambda_=0.01):
    """Iteratively solves Inverse Kinematics using Weighted Damped Least Squares (DLS)."""
    q = np.array(initial_guess, dtype=float)
    for _ in range(max_iter):
        pos_curr, _ = fk(q)
        error = target_pos - pos_curr
        if np.linalg.norm(error) < tol: break
        Jv = jacobian_pos(q)
        
        # Weighted Damped Least Squares equation
        A = Jv.T @ Jv + (lambda_**2) * JOINT_PENALTY_WEIGHTS
        dq = np.linalg.solve(A, Jv.T @ error)
        
        q = q + alpha * dq
    return q

def dh_to_sim(q): return JOINT_SIGN * q

def resample_to_n(configs, n):
    """Stretches an array of joint angles to perfectly match the simulation time steps."""
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

# ---------------------------------------------------------------------
# Quintic Polynomial Trajectory (S-Curves)
# A simple linear movement (`1.0 - t + t`) causes infinite jerk at the start and end,
# damaging real robotic gearboxes. We use a 5th-degree polynomial: `6t^5 - 15t^4 + 10t^3`.
# This forces the velocity AND acceleration to be exactly zero at the start and end of the move.
# ---------------------------------------------------------------------
def build_joint_trajectory(q_start, q_target, n_steps=IK_STEPS):
    """Draws a smooth line between angles in Joint Space."""
    configs = []
    for k in range(n_steps):
        t = k / (n_steps - 1)
        s = 6*t**5 - 15*t**4 + 10*t**3
        configs.append((1.0 - s) * q_start + s * q_target)
    return configs

def build_cartesian_trajectory(pos_start, pos_target, q_seed, n_ik=IK_STEPS):
    """Draws a perfect straight 3D line in the real world (Cartesian Space)."""
    configs = []
    q = q_seed.copy()
    for k in range(n_ik):
        t   = k / (n_ik - 1)
        s   = 6*t**5 - 15*t**4 + 10*t**3            
        pos = (1.0 - s) * pos_start + s * pos_target
        q   = inverse_kinematics_pos(pos, q)
        configs.append(q.copy())
    return configs

# =====================================================================
# 4. COPPELIA EXECUTION HELPERS
# =====================================================================
TARGET_ARM   = '/yaskawa'
STR_SIGNAL   = TARGET_ARM + '_executedMovId'
_exec_mov_id = 'notReady'

def wait_for_movement(sim, move_id, cup_handle, ee_handle, joint_handles, timeout=60.0):
    """Pauses Python execution until CoppeliaSim finishes the requested physical movement."""
    global _exec_mov_id
    _exec_mov_id = 'notReady'
    t0 = time.time()
    while _exec_mov_id != move_id:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Movement '{move_id}' timed out after {timeout} s")
        s = sim.getStringSignal(STR_SIGNAL)
        if s is not None:
            _exec_mov_id = s.decode() if isinstance(s, bytes) else s
        log_simulation_data(sim, cup_handle, ee_handle, joint_handles)
        time.sleep(0.05)

def dispatch(sim, configs_sim, times, move_id, gripper_vel=0.0):
    """Packages the trajectory data and sends it over ZMQ to the Lua script inside the simulator."""
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
# 5. MAIN EXECUTION (MAIN THREAD)
# =====================================================================
def main():
    print("--- Connecting main thread to CoppeliaSim ---")
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    # Get Handles
    joint_handles = [sim.getObject(f'{TARGET_ARM}/joint{i+1}') for i in range(6)]
    try:    cup_handle = sim.getObject('./Cup')
    except: cup_handle = -1; print("Warning: './Cup' not found.")
    try:    ee_handle = sim.getObject('./gripperEF')
    except: ee_handle = -1; print("Warning: './gripperEF' not found.")

    if cup_handle != -1:
        cup_pos = sim.getObjectPosition(cup_handle, sim.handle_world)
        PLACE_POSITIONS = [
            np.array([cup_pos[0] - 0.15, cup_pos[1], cup_pos[2]]),
            np.array(cup_pos),
        ]
    else:
        PLACE_POSITIONS = [np.array([0,0,0])]

    # Calculate Trajectories First
    print("\n--- Phase 3: Calculating Placing Trajectories ---")
    
    q_sim_current = np.array([sim.getJointPosition(h) for h in joint_handles])
    q_dh_current  = JOINT_SIGN * q_sim_current
    
    q_home_dh = q_dh_current.copy() 
    
    q_current_placing = q_dh_current.copy()
    all_placing_trajectories = []

    # Pre-compute all placement movements
    for i, target_pos in enumerate(PLACE_POSITIONS):
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

    print("\n--- Phase 4: Calculating Lift Trajectory ---")
    target_pos_lift = np.array([0.64, 0.0, 0.70])
    q_target_lift = inverse_kinematics_pos(target_pos_lift, q_current_placing)
    lift_configs_dh = build_joint_trajectory(q_current_placing, q_target_lift, n_steps=IK_STEPS)
    lift_sim = [dh_to_sim(q) for q in lift_configs_dh]
    
    N_LIFT = int(LIFT_DUR / DT)
    lift_trajectory = {
        'configs': resample_to_n(lift_sim, N_LIFT),
        'times':   [k * DT for k in range(N_LIFT)],
        'id':      'waypoint_path_lift'
    }

    # START SIMULATION
    print("\n--- Starting Simulation ---")
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_advancing_running:
        time.sleep(0.1)

    # ==========================================================
    # START VISION BACKGROUND THREAD
    # ==========================================================
    # Here we spawn the background process. It runs `vision_thread_worker`.
    # `daemon=True` means if the main program crashes, this thread will close automatically.
    stop_vision_event = threading.Event()
    vision_thread = threading.Thread(target=vision_thread_worker, args=(stop_vision_event,), daemon=True)
    vision_thread.start()
    # ==========================================================

    global _current_gripper_cmd
    _current_gripper_cmd = G_CLOSE

    # Execution Phase
    try:
        print(f"\n--- Executing Placing Trajectories ---")
        for i, traj in enumerate(all_placing_trajectories):
            dispatch(sim, traj['configs'], traj['times'], traj['id'], gripper_vel=G_HOLD)
            wait_for_movement(sim, traj['id'], cup_handle, ee_handle, joint_handles)
            time.sleep(0.5)

        print("\n--- Closing Gripper ---")
        _current_gripper_cmd = G_CLOSE
        N_CLOSE = int(1.0 / DT)
        final_q = all_placing_trajectories[-1]['configs'][-1]
        close_configs = [final_q] * N_CLOSE
        close_times   = [k * DT for k in range(N_CLOSE)]
        
        dispatch(sim, close_configs, close_times, 'waypoint_path_close', gripper_vel=G_CLOSE)
        wait_for_movement(sim, 'waypoint_path_close', cup_handle, ee_handle, joint_handles)
        time.sleep(0.5)

        print("\n--- Lifting to Final Position ---")
        dispatch(sim, lift_trajectory['configs'], lift_trajectory['times'], lift_trajectory['id'], gripper_vel=G_CLOSE)
        wait_for_movement(sim, lift_trajectory['id'], cup_handle, ee_handle, joint_handles)
        time.sleep(0.5)
        
        q_curr = q_target_lift.copy()

        # ==========================================================
        # PHASE 5 - VISUAL SERVOING (Using Memorized Target)
        # Image-Based Visual Servoing (IBVS)
        # Instead of moving to pre-programmed coordinates, the robot looks at the camera feed,
        # calculates the error between the target pixel and current pixel, and converts that 
        # pixel error (dx, dy) into a physical world movement (dx_world, dy_world).
        # It loops this process (Proportional Control) until the error is near zero.
        # ==========================================================
        print("\n--- Phase 5: Vision-Based Centering ---")
        
        # Safely grab the saved coordinate from the Vision Thread
        with vision_state_lock:
            saved_target = vision_state['target_center']
            
        if saved_target is None:
            print("  [Error] Could not execute visual servoing.")
        else:
            print(f"  [Vision] Using memorized target center: {saved_target}")
            
            # Calibration constants for the Visual Servoing controller
            PIXEL_TO_METER = 0.0005  # Approximate ratio: 1 pixel error = 0.5mm movement
            TOLERANCE_PX = 15        # Stop adjusting when within 15 pixels of target
            MAX_ADJUSTMENTS = 20     # Fail-safe to prevent infinite looping
            STEP_DURATION = 0.6      # Time given for the physical arm to make the micro-adjustment
            CAMERA_AXIS_MAPPING = (1.0, 1.0) 

            for step in range(MAX_ADJUSTMENTS):
                # Check live position of ArUco 0
                with vision_state_lock:
                    curr = vision_state['id0_pos']
                
                if curr is None:
                    time.sleep(1.0)
                    continue
                    
                err_x = saved_target[0] - curr[0]
                err_y = saved_target[1] - curr[1]
                dist = math.hypot(err_x, err_y) # Pythagorean theorem for total error distance
                
                print(f"  Step {step+1}: Error = {dist:.1f}px (dx={err_x}, dy={err_y})")
                
                if dist <= TOLERANCE_PX:
                    break
                    
                # Convert 2D pixel error into 3D Cartesian coordinates
                dx_world = err_x * PIXEL_TO_METER * CAMERA_AXIS_MAPPING[0]
                dy_world = err_y * PIXEL_TO_METER * CAMERA_AXIS_MAPPING[1]
                
                p_curr, r_curr = fk(q_curr)
                p_new = p_curr + np.array([dx_world, dy_world, 0.0])
                # Solve IK for the new micro-adjustment
                q_new = inverse_kinematics_pos(p_new, q_curr)
                
                step_configs = build_joint_trajectory(q_curr, q_new, n_steps=10)
                step_sim = [dh_to_sim(q) for q in step_configs]
                
                N_STEP = int(STEP_DURATION / DT)
                resampled = resample_to_n(step_sim, N_STEP)
                times = [k * DT for k in range(N_STEP)]
                mov_id = f'vision_center_step_{step}'
                
                dispatch(sim, resampled, times, mov_id, gripper_vel=G_CLOSE)
                wait_for_movement(sim, mov_id, cup_handle, ee_handle, joint_handles)
                
                q_curr = q_new.copy()
                time.sleep(0.2) 

        # ==========================================================
        # PHASE 6 - ROTATE ONLY JOINT 4
        # Explicit manipulation of individual joints, ignoring end-effector XYZ goals.
        # ==========================================================
        print("\n--- Phase 6: Rotating Joint 4 ---")
        
        q_start_rot = q_curr.copy()
        q_target_rot = q_start_rot.copy()
        q_target_rot[3] = math.pi  
        
        rot_configs_dh = build_joint_trajectory(q_start_rot, q_target_rot, n_steps=IK_STEPS)
        rot_configs_sim = [dh_to_sim(q) for q in rot_configs_dh]
        
        ROT_DUR = 2.0
        N_STEPS_ROT = int(ROT_DUR / DT)
        times_rot = [k * DT for k in range(N_STEPS_ROT)]
        configs_sim_rot = resample_to_n(rot_configs_sim, N_STEPS_ROT)
        move_id_rot = 'waypoint_path_rot_placing'
        
        dispatch(sim, configs_sim_rot, times_rot, move_id_rot, gripper_vel=G_CLOSE)
        wait_for_movement(sim, move_id_rot, cup_handle, ee_handle, joint_handles)

        # ==========================================================
        # PHASE 7 - ROTATE ONLY JOINT 5
        # ==========================================================
        print("\n--- Phase 7: Rotating Joint 5 ---")
        
        q_start_pitch = q_target_rot.copy()
        q_target_pitch = q_start_pitch.copy()
        q_target_pitch[4] = math.pi / 8   
        
        pitch_configs_dh = build_joint_trajectory(q_start_pitch, q_target_pitch, n_steps=IK_STEPS)
        pitch_configs_sim = [dh_to_sim(q) for q in pitch_configs_dh]
        
        PITCH_DUR = 2.0
        N_STEPS_PITCH = int(PITCH_DUR / DT)
        times_pitch = [k * DT for k in range(N_STEPS_PITCH)]
        configs_sim_pitch = resample_to_n(pitch_configs_sim, N_STEPS_PITCH)
        move_id_pitch = 'waypoint_path_pitch'
        
        dispatch(sim, configs_sim_pitch, times_pitch, move_id_pitch, gripper_vel=G_CLOSE)
        wait_for_movement(sim, move_id_pitch, cup_handle, ee_handle, joint_handles)

        # ==========================================================
        # PHASE 8 - MOVE DOWN TO Z = +0.540 (CARTESIAN)
        # ==========================================================
        print("\n--- Phase 8: Moving Down to Z = +0.540 (Cartesian Space) ---")
        
        current_pos, _ = fk(q_target_pitch)
        target_pos_down = np.array([current_pos[0] + 0.05, current_pos[1], 0.540])
        
        down_configs_dh = build_cartesian_trajectory(current_pos, target_pos_down, q_target_pitch, n_ik=IK_STEPS)
        down_configs_sim = [dh_to_sim(q) for q in down_configs_dh]
        
        DOWN_DUR = 3.0
        N_STEPS_DOWN = int(DOWN_DUR / DT)
        times_down = [k * DT for k in range(N_STEPS_DOWN)]
        configs_sim_down = resample_to_n(down_configs_sim, N_STEPS_DOWN)
        move_id_down = 'waypoint_path_down'
        
        dispatch(sim, configs_sim_down, times_down, move_id_down, gripper_vel=G_CLOSE)
        wait_for_movement(sim, move_id_down, cup_handle, ee_handle, joint_handles)
        print("  Move down complete.")
        
        # ---------------------------------------------------------------------
        # !! THE FIX !! Read the EXACT live positions from the simulator
        # Because the physics engine (gravity, collision, friction) runs slightly 
        # differently than perfect mathematical FK/IK, over a long sequence, they drift.
        # By pulling the live `q_sim_actual`, we resync the math to the physics, preventing
        # the robot from "jumping" violently to correct the math error when opening the gripper.
        # ---------------------------------------------------------------------
        q_sim_actual = np.array([sim.getJointPosition(h) for h in joint_handles])
        q_dh_actual = JOINT_SIGN * q_sim_actual

        # ==========================================================
        # PHASE 9 - OPEN GRIPPER (NO SUDDEN MOVEMENT)
        # ==========================================================
        print("\n--- Phase 9: Fully Opening Gripper ---")
        _current_gripper_cmd = G_OPEN
        OPEN_DUR = 1.5   
        N_OPEN = int(OPEN_DUR / DT)
        
        # Use the actual live positions read from Phase 8. 
        # This is guaranteed not to jump.
        open_configs_sim = [q_sim_actual] * N_OPEN
        open_times   = [k * DT for k in range(N_OPEN)]
        
        dispatch(sim, open_configs_sim, open_times, 'waypoint_path_open', gripper_vel=G_OPEN)
        wait_for_movement(sim, 'waypoint_path_open', cup_handle, ee_handle, joint_handles)
        time.sleep(0.5) 
        print("  Gripper fully open.")

        # ==========================================================
        # PHASE 10 - MOVE BACKWARD ON X-AXIS (CARTESIAN)
        # ==========================================================
        print("\n--- Phase 10: Retreating Backwards on X-axis ---")
        
        current_pos_post_drop, _ = fk(q_dh_actual)
        target_pos_back = np.array([current_pos_post_drop[0] - 0.10, current_pos_post_drop[1], current_pos_post_drop[2]])
        
        back_configs_dh = build_cartesian_trajectory(current_pos_post_drop, target_pos_back, q_dh_actual, n_ik=IK_STEPS)
        back_configs_sim = [dh_to_sim(q) for q in back_configs_dh]
        
        BACK_DUR = 2.0
        N_STEPS_BACK = int(BACK_DUR / DT)
        times_back = [k * DT for k in range(N_STEPS_BACK)]
        configs_sim_back = resample_to_n(back_configs_sim, N_STEPS_BACK)
        move_id_back = 'waypoint_path_back'
        
        dispatch(sim, configs_sim_back, times_back, move_id_back, gripper_vel=G_OPEN)
        wait_for_movement(sim, move_id_back, cup_handle, ee_handle, joint_handles)
        print("  Retreat complete.")
        
        q_dh_after_back = back_configs_dh[-1]

        # ==========================================================
        # PHASE 11 - LIFT UP ON Z-AXIS (CARTESIAN)
        # ==========================================================
        print("\n--- Phase 11: Lifting Up on Z-axis ---")
        
        current_pos_back, _ = fk(q_dh_after_back)
        target_pos_up = np.array([current_pos_back[0], current_pos_back[1], current_pos_back[2] + 0.15])
        
        up_configs_dh = build_cartesian_trajectory(current_pos_back, target_pos_up, q_dh_after_back, n_ik=IK_STEPS)
        up_configs_sim = [dh_to_sim(q) for q in up_configs_dh]
        
        UP_DUR = 2.0
        N_STEPS_UP = int(UP_DUR / DT)
        times_up = [k * DT for k in range(N_STEPS_UP)]
        configs_sim_up = resample_to_n(up_configs_sim, N_STEPS_UP)
        move_id_up = 'waypoint_path_up'
        
        dispatch(sim, configs_sim_up, times_up, move_id_up, gripper_vel=G_OPEN)
        wait_for_movement(sim, move_id_up, cup_handle, ee_handle, joint_handles)
        print("  Lift complete.")
        
        q_dh_after_up = up_configs_dh[-1]

        # ==========================================================
        # PHASE 12 - RETURN TO HOME POSITION (CARTESIAN)
        # ==========================================================
        print("\n--- Phase 12: Return to Home Position (Cartesian) ---")
        
        current_pos_up, _ = fk(q_dh_after_up)
        target_pos_home, _ = fk(q_home_dh)
        
        home_configs_dh = build_cartesian_trajectory(current_pos_up, target_pos_home, q_dh_after_up, n_ik=IK_STEPS)
        home_configs_sim = [dh_to_sim(q) for q in home_configs_dh]
        
        HOME_DUR = 4.0
        N_STEPS_HOME = int(HOME_DUR / DT)
        times_home = [k * DT for k in range(N_STEPS_HOME)]
        configs_sim_home = resample_to_n(home_configs_sim, N_STEPS_HOME)
        move_id_home = 'waypoint_path_home'
        
        dispatch(sim, configs_sim_home, times_home, move_id_home, gripper_vel=G_OPEN)
        wait_for_movement(sim, move_id_home, cup_handle, ee_handle, joint_handles)
        print("  Sequence completely finished.")

        # ==========================================================

        input("\n[Execution Complete] Press Enter to stop the simulation and close windows...")

    except KeyboardInterrupt:
        print("\nProcess manually interrupted by user.")
    finally:
        print("\nShutting down...")
        # ---------------------------------------------------------------------
        # THREAD CLEANUP
        # We trigger `stop_vision_event` to tell the while loop in the Vision Thread to break.
        # `join` waits up to 2 seconds for the thread to safely close its OpenCV windows and exit.
        # ---------------------------------------------------------------------
        stop_vision_event.set()      
        vision_thread.join(timeout=2.0) 
        sim.stopSimulation()
        print("Simulation stopped.")

if __name__ == "__main__":
    main()