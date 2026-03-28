"""
Yaskawa GP8 Pick-and-Place — Fixed grab sequence
-------------------------------------------------
Key fixes vs original:
  1. Gripper OPEN during approach (gripper_vel=+0.04)
  2. After arm reaches cup: close gripper via Lua (arm stays still)
  3. Wait 3s real-time for fingers to close
  4. Attach cup to MicoHand before lifting
  5. Then execute placing trajectories
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import threading
import math
import time

Pi = math.pi

# =====================================================================
# SETTINGS
# =====================================================================
DT        = 0.05
IK_STEPS  = 40
PLACE_DUR = 3.0
OPEN_VEL  =  0.04
CLOSE_VEL = -1.00

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

TARGETS = [
    np.array([-0.032915, -0.887067, 0.486025]),
    np.array([-0.03231, -0.882233, 0.48512]),
    np.array([-0.031694, -0.877359, 0.484205]),
    np.array([-0.031058, -0.872478, 0.483287]),
    np.array([-0.030413, -0.867584, 0.48237]),
    np.array([-0.029721, -0.862713, 0.481447]),
    np.array([-0.028934, -0.857858, 0.480519]),
    np.array([-0.028094, -0.852999, 0.47959]),
    np.array([-0.027244, -0.848145, 0.478659]),
    np.array([-0.026384, -0.843305, 0.477726]),
    np.array([-0.025498, -0.838449, 0.476796]),
    np.array([-0.024621, -0.833601, 0.475863]),
    np.array([-0.023648, -0.828783, 0.474924]),
    np.array([-0.022562, -0.823977, 0.473982]),
    np.array([-0.021453, -0.819179, 0.473037]),
    np.array([-0.020337, -0.814383, 0.472092]),
    np.array([-0.019201, -0.809595, 0.471147]),
    np.array([-0.018061, -0.804801, 0.470201]),
    np.array([-0.016866, -0.800026, 0.469252]),
    np.array([-0.015561, -0.795287, 0.4683]),
    np.array([-0.014181, -0.790559, 0.467345]),
    np.array([-0.012782, -0.78584, 0.466388]),
    np.array([-0.011369, -0.781133, 0.465431]),
    np.array([-0.009935, -0.776417, 0.464475]),
    np.array([-0.008503, -0.771708, 0.463517]),
    np.array([-0.006971, -0.767039, 0.462555]),
    np.array([-0.005307, -0.762407, 0.461592]),
    np.array([-0.003601, -0.757792, 0.460627]),
    np.array([-0.001883, -0.753182, 0.459662]),
    np.array([-0.000145, -0.74858, 0.458696]),
    np.array([0.001601, -0.743976, 0.45773]),
    np.array([0.00339, -0.739393, 0.456762]),
    np.array([0.005307, -0.734868, 0.455793]),
    np.array([0.007324, -0.730382, 0.454823]),
    np.array([0.009373, -0.725911, 0.453853]),
    np.array([0.01143, -0.721446, 0.452882]),
    np.array([0.013506, -0.716983, 0.451912]),
    np.array([0.015588, -0.712521, 0.450941]),
    np.array([0.017769, -0.708118, 0.449968]),
    np.array([0.020095, -0.703784, 0.448999]),
    np.array([0.022485, -0.699487, 0.448029]),
    np.array([0.02489, -0.695197, 0.447059]),
    np.array([0.027313, -0.690917, 0.446089]),
    np.array([0.029748, -0.68664, 0.445118]),
    np.array([0.032217, -0.682381, 0.444148]),
    np.array([0.03482, -0.678212, 0.443181]),
    np.array([0.037557, -0.674124, 0.442218]),
    np.array([0.040327, -0.670062, 0.441256]),
    np.array([0.043105, -0.666005, 0.440292]),
    np.array([0.045906, -0.661954, 0.439331]),
    np.array([0.048711, -0.657911, 0.438367]),
    np.array([0.051608, -0.653932, 0.437408]),
    np.array([0.054644, -0.650062, 0.436456]),
    np.array([0.057756, -0.646252, 0.435509]),
    np.array([0.060887, -0.642457, 0.434562]),
    np.array([0.064028, -0.638676, 0.433614]),
    np.array([0.067197, -0.634895, 0.432668]),
    np.array([0.07038, -0.63113, 0.431722]),
    np.array([0.073671, -0.627479, 0.430783]),
    np.array([0.077104, -0.623945, 0.429857]),
    np.array([0.080566, -0.620448, 0.428932]),
    np.array([0.084036, -0.61695, 0.428015]),
    np.array([0.087525, -0.61347, 0.427143]),
    np.array([0.091028, -0.609992, 0.426359]),
    np.array([0.094594, -0.60659, 0.425705]),
    np.array([0.098285, -0.603325, 0.425226]),
    np.array([0.102058, -0.600156, 0.424948]),
    np.array([0.10585, -0.597005, 0.424894]),
    np.array([0.109646, -0.593869, 0.425085]),
    np.array([0.113475, -0.590743, 0.425548]),
    np.array([0.117305, -0.587619, 0.426302]),
    np.array([0.121218, -0.584619, 0.427374]),
    np.array([0.125259, -0.581778, 0.428789]),
    np.array([0.129322, -0.57898, 0.430541]),
    np.array([0.133397, -0.576187, 0.43264]),
    np.array([0.137485, -0.573406, 0.435091]),
    np.array([0.141587, -0.570641, 0.437898]),
    np.array([0.145725, -0.567935, 0.441072]),
    np.array([0.149943, -0.565361, 0.444624]),
    np.array([0.154232, -0.562897, 0.448549]),
    np.array([0.158533, -0.560453, 0.452834]),
    np.array([0.162834, -0.558022, 0.457469]),
    np.array([0.167167, -0.55561, 0.462454]),
    np.array([0.1715, -0.553205, 0.467775]),
    np.array([0.175881, -0.55091, 0.473434]),
    np.array([0.180343, -0.548753, 0.47943]),
    np.array([0.184831, -0.54666, 0.485735]),
    np.array([0.189323, -0.544573, 0.492328]),
    np.array([0.193825, -0.5425, 0.499195]),
    np.array([0.198346, -0.540453, 0.50632]),
    np.array([0.202882, -0.538442, 0.51369]),
    np.array([0.207461, -0.536552, 0.521294]),
    np.array([0.212092, -0.534765, 0.529117]),
    np.array([0.216733, -0.533013, 0.537123]),
    np.array([0.221371, -0.531269, 0.545286]),
    np.array([0.226038, -0.529546, 0.553594]),
    np.array([0.230705, -0.527842, 0.562016]),
    np.array([0.235392, -0.526217, 0.570541]),
    np.array([0.240124, -0.524708, 0.579158]),
    np.array([0.244877, -0.523259, 0.587829]),
    np.array([0.249633, -0.521824, 0.596525]),
    np.array([0.254389, -0.52041, 0.605221]),
    np.array([0.259178, -0.519019, 0.613902]),
    np.array([0.263962, -0.517655, 0.622537]),
    np.array([0.268762, -0.51639, 0.631115]),
    np.array([0.2736, -0.515214, 0.639619]),
    np.array([0.278439, -0.514071, 0.648011]),
    np.array([0.283278, -0.512938, 0.656264]),
    np.array([0.288132, -0.511828, 0.664365]),
    np.array([0.293001, -0.510738, 0.672289]),
    np.array([0.297865, -0.509707, 0.680018]),
    np.array([0.30275, -0.50877, 0.687545]),
    np.array([0.307654, -0.50788, 0.694844]),
    np.array([0.312556, -0.50701, 0.701888]),
    np.array([0.317454, -0.506159, 0.708659]),
    np.array([0.322381, -0.505328, 0.715152]),
    np.array([0.327303, -0.504519, 0.721342]),
    np.array([0.332222, -0.503785, 0.727228]),
    np.array([0.337163, -0.503127, 0.732804]),
    np.array([0.342107, -0.502494, 0.738047]),
    np.array([0.347048, -0.50187, 0.742945]),
    np.array([0.351993, -0.501271, 0.747496]),
    np.array([0.356962, -0.500691, 0.751696]),
    np.array([0.361916, -0.500154, 0.755539]),
    np.array([0.366872, -0.499686, 0.759034]),
    np.array([0.371843, -0.499258, 0.76218]),
    np.array([0.376813, -0.498846, 0.76497]),
    np.array([0.381777, -0.498454, 0.767409]),
    np.array([0.386759, -0.498081, 0.76951]),
    np.array([0.391746, -0.49772, 0.771278]),
    np.array([0.396718, -0.497418, 0.77273]),
    np.array([0.401698, -0.497179, 0.77389]),
    np.array([0.406682, -0.496957, 0.774762]),
    np.array([0.411663, -0.496742, 0.775365]),
    np.array([0.416647, -0.49655, 0.775725]),
    np.array([0.421649, -0.496375, 0.775869]),
    np.array([0.426637, -0.496232, 0.775823]),
    np.array([0.431619, -0.496148, 0.775627]),
    np.array([0.436608, -0.496102, 0.775315]),
    np.array([0.441597, -0.496067, 0.774919]),
    np.array([0.446582, -0.496049, 0.774479]),
    np.array([0.451577, -0.496049, 0.774036]),
    np.array([0.456582, -0.496059, 0.773597]),
    np.array([0.461568, -0.496118, 0.773161]),
    np.array([0.466553, -0.496234, 0.772737]),
    np.array([0.471543, -0.496365, 0.772319]),
    np.array([0.47653, -0.496503, 0.7719]),
    np.array([0.481517, -0.496662, 0.771484]),
    np.array([0.48652, -0.496839, 0.771074]),
    np.array([0.491513, -0.497041, 0.770666]),
    np.array([0.496492, -0.497299, 0.770266]),
    np.array([0.501472, -0.4976, 0.769877]),
    np.array([0.506453, -0.497912, 0.76949]),
    np.array([0.511431, -0.498238, 0.769104]),
    np.array([0.516418, -0.498582, 0.768723]),
    np.array([0.521414, -0.498939, 0.768345]),
    np.array([0.526386, -0.499344, 0.767972]),
    np.array([0.531353, -0.499815, 0.767613]),
    np.array([0.536321, -0.500314, 0.767261]),
    np.array([0.541286, -0.500821, 0.76691]),
    np.array([0.546247, -0.501349, 0.766561]),
]

PLACE_POSITIONS = [
    np.array([0.865, 0.0,  0.700]),
    np.array([0.015, 0.5,  0.700]),
]

TARGET_DUR = [i * DT for i in range(len(TARGETS))][-1] + 1.5


# =====================================================================
# KINEMATICS
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

def inverse_kinematics_pos(target_pos, initial_guess,
                            max_iter=300, tol=1e-4, alpha=0.5, lambda_=0.01):
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
# TRAJECTORY
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
        print(f"  IK waypoint {k+1:3d}/{n_ik}  pos={np.round(pos,4)}  "
              f"q_deg={np.round(np.degrees(q),2).tolist()}")
    return configs

def build_joint_trajectory(q_start, q_target, n_steps=IK_STEPS):
    configs = []
    for k in range(n_steps):
        t = k / (n_steps - 1)
        s = 6*t**5 - 15*t**4 + 10*t**3
        configs.append((1.0 - s) * q_start + s * q_target)
    return configs


# =====================================================================
# COPPELIA HELPERS
# =====================================================================

TARGET_ARM   = '/yaskawa'
STR_SIGNAL   = TARGET_ARM + '_executedMovId'
_exec_mov_id = 'notReady'

SCRIPT_TARGETS = ['yaskawa', '/yaskawa', 'yaskawa#0']
_script_target = None

def safe_get(sim, path, fallback):
    try:
        h = sim.getObject(path)
        print(f"  [handle] '{path}' -> {h}")
        return h
    except Exception:
        print(f"  [handle] '{path}' failed, using handle {fallback}")
        return fallback

def find_script_target(sim):
    global _script_target
    md_probe = {
        'id': '_probe', 'type': 'pts', 'times': [0.0, DT],
        'j1': [0.0]*2, 'j2': [0.0]*2, 'j3': [0.0]*2,
        'j4': [0.0]*2, 'j5': [0.0]*2, 'j6': [0.0]*2,
        'gripper': [0.0]*2,
    }
    for t in SCRIPT_TARGETS:
        try:
            sim.callScriptFunction(
                f'remoteApi_movementDataFunction@{t}',
                sim.scripttype_childscript, md_probe)
            print(f"  [script] Working target: '{t}'")
            _script_target = t
            return
        except Exception:
            pass
    raise Exception("Cannot find yaskawa Lua script. Check scene hierarchy.")

def wait_for_movement(sim, move_id, timeout=60.0):
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

def dispatch(sim, configs_sim, times, move_id, gripper_vel=0.0):
    assert len(configs_sim) == len(times), "configs_sim length != times length"
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
        f'remoteApi_movementDataFunction@{_script_target}',
        sim.scripttype_childscript, md)
    sim.callScriptFunction(
        f'remoteApi_executeMovement@{_script_target}',
        sim.scripttype_childscript, move_id)


# =====================================================================
# MAIN
# =====================================================================

grab_request = False

def grab_listener():
    global grab_request
    input("\nPress ENTER anytime to close gripper...\n")
    grab_request = True

if __name__ == "__main__":
    print("--- Connecting to CoppeliaSim ---")
    client = RemoteAPIClient()
    sim    = client.getObject('sim')

    print("\n--- Resolving object handles ---")
    cup_handle        = safe_get(sim, '/conveyorSystem/Cup',          162)
    microhand_handle  = safe_get(sim, '/yaskawa/MicoHand',             37)

    joint_handles = [sim.getObject(f'{TARGET_ARM}/joint{i+1}') for i in range(6)]

    # Pre-compute grab trajectory
    N_STEPS = int(TARGET_DUR / DT)
    times   = [i * DT for i in range(N_STEPS)]

    q_sim_current = np.array([sim.getJointPosition(h) for h in joint_handles])
    q_dh_current  = JOINT_SIGN * q_sim_current
    pos_start, _  = fk(q_dh_current)

    print("\n--- Phase 1: Cartesian approach ---")
    ik_configs_dh = [q_dh_current.copy()]
    q_seed = q_dh_current.copy()
    approach_configs = build_cartesian_trajectory(pos_start, TARGETS[0], q_seed)
    ik_configs_dh.extend(approach_configs)
    q_seed = approach_configs[-1].copy()

    print(f"\n--- Phase 2: IK for {len(TARGETS)-1} waypoints ---")
    for idx in range(1, len(TARGETS)):
        q_ik = inverse_kinematics_pos(TARGETS[idx], q_seed)
        ik_configs_dh.append(q_ik.copy())
        q_seed = q_ik.copy()
        if idx % 10 == 0 or idx == len(TARGETS) - 1:
            print(f"  Solved {idx}/{len(TARGETS)-1}...")

    ik_configs_sim = [dh_to_sim(q) for q in ik_configs_dh]
    configs_sim    = resample_to_n(ik_configs_sim, N_STEPS)

    print("\n--- Phase 3: Placing trajectories ---")
    q_after_grab = ik_configs_dh[-1]
    all_placing  = []
    for i, pos in enumerate(PLACE_POSITIONS):
        q_target    = inverse_kinematics_pos(pos, q_after_grab)
        place_dh    = build_joint_trajectory(q_after_grab, q_target)
        place_sim   = [dh_to_sim(q) for q in place_dh]
        N_PLACE     = max(2, int(PLACE_DUR / DT))
        times_place = [k * DT for k in range(N_PLACE)]
        all_placing.append({
            'configs': resample_to_n(place_sim, N_PLACE),
            'times':   times_place,
            'id':      f'placing_{i+1}',
        })
        q_after_grab = q_target
        print(f"  Place {i+1} computed.")

    # ── Execute (Truncated Trajectory Version) ────────────────────────
    # Your "Golden Joints" in degrees
    Q_TARGET_DEG = np.array([-91.04, 28.69, 7.26, -6.18, 3.62, 0.01])
    Q_TARGET_RAD = np.radians(Q_TARGET_DEG)

    # 1. FIND THE TARGET STEP
    # We look through your pre-computed configs_sim to find the closest match
    distances = [np.linalg.norm(np.array(cfg) - Q_TARGET_RAD) for cfg in configs_sim]
    target_idx = np.argmin(distances) # The exact index where we should stop
    
    # Create a LIMITED trajectory that ends EXACTLY at your target
    configs_limited = configs_sim[:target_idx + 1]
    times_limited   = [i * DT for i in range(len(configs_limited))]

    print(f"\n--- Starting simulation ---")
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_advancing_running:
        time.sleep(0.1)
    
    find_script_target(sim)
    wait_for_movement(sim, 'ready')

    # 2. STEP 1: MOVE EXACTLY TO THE TARGET INDEX
    # Because we only send points up to the target, the robot CANNOT overshoot.
    print(f"\n[1/3] Moving to targeted joint position (Step {target_idx})...")
    dispatch(sim, configs_limited, times_limited, 'precise_approach', gripper_vel=OPEN_VEL)
    
    # Wait for the precise movement to finish
    wait_for_movement(sim, 'precise_approach')
    print(f"  [✔] Robot stopped exactly at target joints.")

    # ── STEP 2: GRAB SEQUENCE (Grip Lock Version) ──────────────────
    print("\n[2/3] Robot stopped. Locking gripper...")
    
    q_final_stop = [sim.getJointPosition(h) for h in joint_handles]
    
    # 1. Force the fingers to close
    # We send a long-duration 'hold' to keep the Lua script busy
    N_HOLD = max(2, int(5.0 / DT)) 
    hold_times = [i * DT for i in range(N_HOLD)]
    hold_cfgs  = [q_final_stop] * N_HOLD
    
    dispatch(sim, hold_cfgs, hold_times, 'lock_grab', gripper_vel=CLOSE_VEL)

    print("  Closing fingers", end="", flush=True)
    for _ in range(30):
        time.sleep(0.1)
        print(".", end="", flush=True)
        # RE-ASSERT the signal every 100ms to fight the Lua reset
        sim.setFloatSignal('gripper_vel', CLOSE_VEL)
    print(" done.")

    # 2. Parent the object while the 'lock_grab' ID is still active
    sim.setObjectParent(cup_handle, microhand_handle, True)
    
    # 3. Small physics settling time
    time.sleep(0.2)
    print("  [✔] Cup attached and fingers locked.")

    # ── STEP 3: DYNAMIC RESUME (No-Gap Transition) ────────────────
    print(f"\n[3/3] Resuming movement...")
    
    q_dh_now = JOINT_SIGN * np.array(q_final_stop)
    
    for i, pos in enumerate(PLACE_POSITIONS):
        print(f"  Placing {i+1} (maintaining grip)...", end="", flush=True)
        
        # Calculate IK while the 'gripper_vel' signal is still being forced
        sim.setFloatSignal('gripper_vel', CLOSE_VEL)
        q_target = inverse_kinematics_pos(pos, q_dh_now)
        
        bridge_dh  = build_joint_trajectory(q_dh_now, q_target)
        bridge_sim = [dh_to_sim(q) for q in bridge_dh]
        
        # DISPATCH NEXT MOVE: The new ID will overwrite the 'lock_grab' ID 
        # but keep the CLOSE_VEL, preventing the "release" twitch.
        dispatch(sim, resample_to_n(bridge_sim, IK_STEPS), 
                 [k * DT for k in range(IK_STEPS)], f'place_{i}', gripper_vel=CLOSE_VEL)
        
        wait_for_movement(sim, f'place_{i}')
        
        # Re-lock signal after the movement ID finishes
        sim.setFloatSignal('gripper_vel', CLOSE_VEL)
        
        q_dh_now = q_target
        print(" done.")

    # ── STEP 4: RELEASE ───────────────────────────────────────────
    print("\n[4/4] Releasing cup.")
    sim.setFloatSignal('gripper_vel', OPEN_VEL)
    sim.setObjectParent(cup_handle, -1, True) 
    
    print("\n--- Sequence Complete. ---")
    sim.stopSimulation()