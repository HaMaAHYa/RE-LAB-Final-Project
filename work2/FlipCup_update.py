"""
Merged Script: Yaskawa GP8 Pick-and-Place + ArUco Vision Tracker
Runs the robot logic in the main thread while processing vision in a background thread.
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
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

# =====================================================================
# 1. SHARED CONFIGURATION & SETUP
# =====================================================================
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
DT = 0.05
IK_STEPS = 40
G_CLOSE = -0.04
G_OPEN  =  0.04
G_HOLD  =  0.0
PLACE_DUR = 3.0
LIFT_DUR  = 3.0

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
WRIST_OFFSET = T_TOOL[2, 3]

APPROACH_DUR = IK_STEPS * DT
TARGET_DUR   = APPROACH_DUR

# Data Logging Globals
DATA_LOG = {
    't': [],
    'cup_pos': [], 'cup_vel': [], 'cup_ori': [], 'cup_omega': [],
    'ee_pos':  [], 'ee_vel':  [], 'ee_ori':  [], 'ee_omega':  [],
    'joint_pos': [], 'gripper_cmd': [],
}
_current_gripper_cmd  = G_HOLD
_dynamic_gripper_list = None
_dynamic_gripper_t0   = None


# =====================================================================
# 2. ARUCO VISION THREAD FUNCTIONS
# =====================================================================
def get_handle(sim, possible_names, fallback_id=None):
    for name in possible_names:
        try:
            handle = sim.getObject(name)
            if handle != -1: return handle
        except:
            continue
    return fallback_id

def get_image(sim, sensor_handle):
    try:
        img, resolution = sim.getVisionSensorImg(sensor_handle)
        if not img or resolution[0] == 0: return None
        arr = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
        return cv2.flip(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), 0)
    except: 
        return None

def detect_and_annotate(frame, detector):
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
            
            meaning = ARUCO_MEANINGS.get(mid, f"ID:{mid}")
            cv2.putText(frame, f"{meaning} ({cx},{cy})", (cx - 40, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, found_markers

def vision_thread_worker(stop_event):
    """
    This function runs in a completely separate thread.
    It creates its own ZMQ client to prevent cross-thread socket crashes.
    """
    print("[Vision Thread] Connecting dedicated client...")
    client_vis = RemoteAPIClient()
    sim_vis = client_vis.getObject('sim')
    
    sensor_handle = get_handle(sim_vis, ['./Vision_sensor', 'Vision_sensor'], fallback_id=384)
    if sensor_handle is None or sensor_handle == -1:
        print("[Vision Thread] ✘ Error: Could not find Vision Sensor. Thread stopping.")
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    WINDOW_NAME = "RE-LAB ArUco Tracking"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("[Vision Thread] Started Tracking Window.")
    
    # Run until the main thread tells us to stop
    while not stop_event.is_set():
        frame = get_image(sim_vis, sensor_handle)
        if frame is not None:
            annotated, marker_centers = detect_and_annotate(frame, detector)
            
            # Optional: Calculate center if 4 markers are found
            valid_pts = [marker_centers[m] for m in [1, 2, 3, 4] if m in marker_centers]
            if len(valid_pts) == 4:
                avg_x = sum(p[0] for p in valid_pts) // 4
                avg_y = sum(p[1] for p in valid_pts) // 4
                cv2.circle(annotated, (avg_x, avg_y), 5, (0, 0, 255), -1)
                cv2.putText(annotated, "CENTER", (avg_x + 10, avg_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow(WINDOW_NAME, annotated)

        # Process GUI events and check for manual quit 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Vision Thread] Manual quit requested.")
            break
            
    cv2.destroyAllWindows()
    print("[Vision Thread] Closed gracefully.")


# =====================================================================
# 3. ROBOT KINEMATICS & TRAJECTORY FUNCTIONS (MAIN THREAD)
# =====================================================================
def log_simulation_data(sim, cup_handle, ee_handle, joint_handles):
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
    if joint_handles:
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

def jacobian_pos(q, fd_eps=1e-7):
    Jv = np.zeros((3, 6))
    for i in range(6):
        qp, qm = q.copy(), q.copy()
        qp[i] += fd_eps; qm[i] -= fd_eps
        pp, _ = fk(qp); pm, _ = fk(qm)
        Jv[:, i] = (pp - pm) / (2 * fd_eps)
    return Jv

JOINT_PENALTY_WEIGHTS = np.diag([0.1, 0.1, 0.1, 0.1, 0.5, 0.4])

def inverse_kinematics_pos(target_pos, initial_guess, max_iter=300, tol=1e-4, alpha=0.5, lambda_=0.01):
    q = np.array(initial_guess, dtype=float)
    for _ in range(max_iter):
        pos_curr, _ = fk(q)
        error = target_pos - pos_curr
        if np.linalg.norm(error) < tol: break
        Jv = jacobian_pos(q)
        A = Jv.T @ Jv + (lambda_**2) * JOINT_PENALTY_WEIGHTS
        dq = np.linalg.solve(A, Jv.T @ error)
        q = q + alpha * dq
    return q

def dh_to_sim(q):
    return JOINT_SIGN * q

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

def build_joint_trajectory(q_start, q_target, n_steps=IK_STEPS):
    configs = []
    for k in range(n_steps):
        t = k / (n_steps - 1)
        s = 6*t**5 - 15*t**4 + 10*t**3
        configs.append((1.0 - s) * q_start + s * q_target)
    return configs

# =====================================================================
# 4. COPPELIA EXECUTION HELPERS
# =====================================================================
TARGET_ARM   = '/yaskawa'
STR_SIGNAL   = TARGET_ARM + '_executedMovId'
_exec_mov_id = 'notReady'

def wait_for_movement(sim, move_id, cup_handle, ee_handle, joint_handles, timeout=60.0):
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
        PLACE_POSITIONS = [np.array([0,0,0])] # Failsafe

    # Calculate Trajectories First
    print("\n--- Phase 3: Calculating Placing Trajectories ---")
    q_sim_current = np.array([sim.getJointPosition(h) for h in joint_handles])
    q_dh_current  = JOINT_SIGN * q_sim_current
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
    stop_vision_event = threading.Event()
    vision_thread = threading.Thread(target=vision_thread_worker, args=(stop_vision_event,), daemon=True)
    vision_thread.start()
    # ==========================================================

    global _current_gripper_cmd
    _current_gripper_cmd = G_CLOSE

    # Execution Phase
    try:
        print(f"\n--- Executing {len(all_placing_trajectories)} Placing Trajectories ---")
        for i, traj in enumerate(all_placing_trajectories):
            print(f"  Sending Placing {i+1} (Duration: {PLACE_DUR}s)...")
            dispatch(sim, traj['configs'], traj['times'], traj['id'], gripper_vel=G_HOLD)
            wait_for_movement(sim, traj['id'], cup_handle, ee_handle, joint_handles)
            print("  Done.")
            time.sleep(0.5)

        print("\n--- Closing Gripper ---")
        _current_gripper_cmd = G_CLOSE
        N_CLOSE = int(1.0 / DT)
        final_q = all_placing_trajectories[-1]['configs'][-1]
        close_configs = [final_q] * N_CLOSE
        close_times   = [k * DT for k in range(N_CLOSE)]
        
        dispatch(sim, close_configs, close_times, 'waypoint_path_close', gripper_vel=G_CLOSE)
        wait_for_movement(sim, 'waypoint_path_close', cup_handle, ee_handle, joint_handles)
        print("  Gripper closed.")
        time.sleep(0.5)

        print("\n--- Lifting to Final Position ---")
        dispatch(sim, lift_trajectory['configs'], lift_trajectory['times'], lift_trajectory['id'], gripper_vel=G_CLOSE)
        wait_for_movement(sim, lift_trajectory['id'], cup_handle, ee_handle, joint_handles)
        print("  Lift complete.")
        time.sleep(0.5)

        input("\n[Execution Complete] Press Enter to stop the simulation and close windows...")

    except KeyboardInterrupt:
        print("\nProcess manually interrupted by user.")
    finally:
        # CLEANUP: Stop the background thread, stop the sim
        print("\nShutting down...")
        stop_vision_event.set()      # Tell vision loop to stop
        vision_thread.join(timeout=2.0) # Wait for thread to close gracefully
        
        sim.stopSimulation()
        print("Simulation stopped.")

if __name__ == "__main__":
    main()
