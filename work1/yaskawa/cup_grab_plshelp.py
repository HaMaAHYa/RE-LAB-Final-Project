from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math
import time

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
    T = np.eye(4)
    for i, (a, alpha, d, theta_off) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(a, alpha, d, q[i] + theta_off)
    T = T @ T_TOOL
    return T[:3, 3].copy(), T[:3, :3].copy()

# =====================================================================
# 3.  JACOBIAN  (position rows only — top 3 of full 6x6)
# =====================================================================
FD_EPS = 1e-7

def jacobian_pos(q):
    Jv = np.zeros((3, 6))
    for i in range(6):
        q_p, q_m = q.copy(), q.copy()
        q_p[i] += FD_EPS
        q_m[i] -= FD_EPS
        p_p, _ = fk(q_p)
        p_m, _ = fk(q_m)
        Jv[:, i] = (p_p - p_m) / (2 * FD_EPS)
    return Jv

# =====================================================================
# 4.  MATH HELPERS
# =====================================================================
def R_to_rpy(R):
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    if abs(np.cos(pitch)) < 1e-6:
        roll = 0.0
        yaw  = np.arctan2(-R[1,2], R[1,1])
    else:
        roll = np.arctan2(R[2,1], R[2,2])
        yaw  = np.arctan2(R[1,0], R[0,0])
    return roll, pitch, yaw

# =====================================================================
# 5.  IK SOLVER
# =====================================================================
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

# =====================================================================
# 6.  SIGN CONVERSION
# =====================================================================
def dh_to_sim(q):
    return JOINT_SIGN * q

# =====================================================================
# 7.  TRAJECTORY HELPERS
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

IK_STEPS = 40
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

# =====================================================================
# 8.  COPPELIA HELPERS
# =====================================================================
TARGET_ARM   = '/yaskawa'
STR_SIGNAL   = TARGET_ARM + '_executedMovId'
_exec_mov_id = 'notReady'

# callScriptFunction requires the object name WITHOUT a leading slash.
# We auto-probe all candidates and keep the first that works.
SCRIPT_TARGETS = [
    'yaskawa',    # bare name — most common in CoppeliaSim 4.x
    '/yaskawa',   # full alias path (some builds accept this)
    'yaskawa#0',  # legacy suffixed name
]
_script_target = None  # resolved lazily on first dispatch

def _find_script_target(sim):
    # Send a harmless single-point probe to find the working target name.
    md_probe = {
        'id': '_probe', 'type': 'pts', 'times': [0.0],
        'j1': [0.0], 'j2': [0.0], 'j3': [0.0],
        'j4': [0.0], 'j5': [0.0], 'j6': [0.0],
        'gripper': [0.0],
    }
    for target in SCRIPT_TARGETS:
        try:
            sim.callScriptFunction(
                f'remoteApi_movementDataFunction@{target}',
                sim.scripttype_childscript, md_probe)
            print(f'  [script] Working target: "{target}"')
            return target
        except Exception:
            pass
    return None

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
    global _script_target
    assert len(configs_sim) == len(times), \
        f"configs_sim length {len(configs_sim)} != times length {len(times)}"
    if _script_target is None:
        _script_target = _find_script_target(sim)
        if _script_target is None:
            raise Exception(
                'Cannot find yaskawa Lua script. '
                'Tried: ' + str(SCRIPT_TARGETS) + '. '
                'Check the object name in the CoppeliaSim scene hierarchy.'
            )
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
# 9.  MAIN
# =====================================================================

# ── Connect ────────────────────────────────────────────────────────
client = RemoteAPIClient()
sim    = client.getObject('sim')

# ── Object handles — using safe fallback to integer handles ───────
# Scene uses new alias system. Path lookups tried first;
# if they fail, integer handles from Part_Name.txt are used directly.
#   93  = gripperEF        (direct child of yaskawa, NOT under MicoHand)
#   37  = MicoHand#0       alias /yaskawa/MicoHand
#  162  = Cup#0            alias /conveyorSystem/Cup
#  170  = cup_pose#0       pose dummy inside Cup (accurate grab centre)

def safe_get(sim, path, fallback):
    try:
        h = sim.getObject(path)
        print(f"  [handle] '{path}' -> {h}")
        return h
    except Exception:
        print(f"  [handle] '{path}' failed, using hard-coded handle {fallback}")
        return fallback

gripper_ef_handle = safe_get(sim, '/yaskawa/gripperEF',            93)
cup_handle        = safe_get(sim, '/conveyorSystem/Cup',          162)
cup_pose_handle   = safe_get(sim, '/conveyorSystem/Cup/cup_pose', 170)
microhand_handle  = safe_get(sim, '/yaskawa/MicoHand',             37)

TCP     = sim.getObjectPosition(gripper_ef_handle, -1)
cup_pos = sim.getObjectPosition(cup_pose_handle, -1)   # use cup_pose#0 for accurate centre

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
    np.array([0.084036, -0.61695, 0.428007]),
    np.array([0.087525, -0.61347, 0.427083]),
    np.array([0.091028, -0.609992, 0.42616]),
    np.array([0.094594, -0.60659, 0.425242]),
    np.array([0.098285, -0.603325, 0.424339]),
    np.array([0.102058, -0.600156, 0.423445]),
    np.array([0.10585, -0.597005, 0.422554]),
    np.array([0.109646, -0.593869, 0.421661]),
    np.array([0.113475, -0.590743, 0.420771]),
    np.array([0.117305, -0.587619, 0.419881]),
    np.array([0.121218, -0.584619, 0.419002]),
    np.array([0.125259, -0.581778, 0.418144]),
    np.array([0.129322, -0.57898, 0.417289]),
    np.array([0.133397, -0.576187, 0.416436]),
    np.array([0.137485, -0.573406, 0.415583]),
    np.array([0.141587, -0.570641, 0.41473]),
    np.array([0.145725, -0.567935, 0.413884]),
    np.array([0.149943, -0.565361, 0.413055]),
    np.array([0.154232, -0.562897, 0.41224]),
    np.array([0.158533, -0.560453, 0.411428]),
    np.array([0.162834, -0.558022, 0.410615]),
    np.array([0.167167, -0.55561, 0.409807]),
    np.array([0.1715, -0.553205, 0.408999]),
    np.array([0.175881, -0.55091, 0.408202]),
    np.array([0.180343, -0.548753, 0.407428]),
    np.array([0.184831, -0.54666, 0.40666]),
    np.array([0.189323, -0.544573, 0.405893]),
    np.array([0.193825, -0.5425, 0.405127]),
    np.array([0.198346, -0.540453, 0.404364]),
    np.array([0.202882, -0.538442, 0.403607]),
    np.array([0.207461, -0.536552, 0.402865]),
    np.array([0.212092, -0.534765, 0.402141]),
    np.array([0.216733, -0.533013, 0.40142]),
    np.array([0.221371, -0.531269, 0.400698]),
    np.array([0.226038, -0.529546, 0.399983]),
    np.array([0.230705, -0.527842, 0.399267]),
    np.array([0.235392, -0.526217, 0.398561]),
    np.array([0.240124, -0.524708, 0.397877]),
    np.array([0.244877, -0.523259, 0.3972]),
    np.array([0.249633, -0.521824, 0.396525]),
    np.array([0.254389, -0.52041, 0.39585]),
    np.array([0.259178, -0.519019, 0.395183]),
    np.array([0.263962, -0.517655, 0.394517]),
    np.array([0.268762, -0.51639, 0.393864]),
    np.array([0.2736, -0.515214, 0.39323]),
    np.array([0.278439, -0.514071, 0.392599]),
    np.array([0.283278, -0.512938, 0.391967]),
    np.array([0.288132, -0.511828, 0.391341]),
    np.array([0.293001, -0.510738, 0.390718]),
    np.array([0.297865, -0.509707, 0.390101]),
    np.array([0.30275, -0.50877, 0.389501]),
    np.array([0.307654, -0.50788, 0.388912]),
    np.array([0.312556, -0.50701, 0.388323]),
    np.array([0.317454, -0.506159, 0.387734]),
    np.array([0.322381, -0.505328, 0.387154]),
    np.array([0.327303, -0.504519, 0.386574]),
    np.array([0.332222, -0.503785, 0.386004]),
    np.array([0.337163, -0.503127, 0.385451]),
    np.array([0.342107, -0.502494, 0.384901]),
    np.array([0.347048, -0.50187, 0.384351]),
    np.array([0.351993, -0.501271, 0.383805]),
    np.array([0.356962, -0.500691, 0.383265]),
    np.array([0.361916, -0.500154, 0.382727]),
    np.array([0.366872, -0.499686, 0.382202]),
    np.array([0.371843, -0.499258, 0.381688]),
    np.array([0.376813, -0.498846, 0.381174]),
    np.array([0.381777, -0.498454, 0.380661]),
    np.array([0.386759, -0.498081, 0.380155]),
    np.array([0.391746, -0.49772, 0.37965]),
    np.array([0.396718, -0.497418, 0.379151]),
    np.array([0.401698, -0.497179, 0.378667]),
    np.array([0.406682, -0.496957, 0.378186]),
    np.array([0.411663, -0.496742, 0.377705]),
    np.array([0.416647, -0.49655, 0.377228]),
    np.array([0.421649, -0.496375, 0.376756]),
    np.array([0.426637, -0.496232, 0.376286]),
    np.array([0.431619, -0.496148, 0.375826]),
    np.array([0.436608, -0.496102, 0.375375]),
    np.array([0.441597, -0.496067, 0.374927]),
    np.array([0.446582, -0.496049, 0.374479]),
    np.array([0.451577, -0.496049, 0.374036]),
    np.array([0.456582, -0.496059, 0.373597]),
    np.array([0.461568, -0.496118, 0.373161]),
    np.array([0.466553, -0.496234, 0.372737]),
    np.array([0.471543, -0.496365, 0.372319]),
    np.array([0.47653, -0.496503, 0.3719]),
    np.array([0.481517, -0.496662, 0.371484]),
    np.array([0.48652, -0.496839, 0.371074]),
    np.array([0.491513, -0.497041, 0.370666]),
    np.array([0.496492, -0.497299, 0.370266]),
    np.array([0.501472, -0.4976, 0.369877]),
    np.array([0.506453, -0.497912, 0.36949]),
    np.array([0.511431, -0.498238, 0.369104]),
    np.array([0.516418, -0.498582, 0.368723]),
    np.array([0.521414, -0.498939, 0.368345]),
    np.array([0.526386, -0.499344, 0.367972]),
    np.array([0.531353, -0.499815, 0.367613]),
    np.array([0.536321, -0.500314, 0.367261]),
    np.array([0.541286, -0.500821, 0.36691]),
    np.array([0.546247, -0.501349, 0.366561]),
]

N = len(TARGETS)

INITIAL_GUESS = np.zeros(6)
JOINT_SIGN    = np.array([+1, +1, -1, +1, -1, +1], dtype=float)
GRIPPER_VEL   = 0.0

DT         = 0.05
TARGET_DUR = [i * DT for i in range(N)][-1] + 2.0

joint_handles = [sim.getObject(f'{TARGET_ARM}/joint{i+1}') for i in range(6)]

N_STEPS = int(TARGET_DUR / DT)
times   = [i * DT for i in range(N_STEPS)]

# ── Read current state ─────────────────────────────────────────────
q_sim_current = np.array([sim.getJointPosition(h) for h in joint_handles])
q_dh_current  = JOINT_SIGN * q_sim_current
pos_start, _  = fk(q_dh_current)

ik_configs_dh = [q_dh_current.copy()]
q_seed        = q_dh_current.copy()

print("\n--- Phase 1: Smooth approach to the first waypoint ---")
approach_configs = build_cartesian_trajectory(
    pos_start  = pos_start,
    pos_target = TARGETS[0],
    q_seed     = q_seed,
    n_ik       = IK_STEPS
)
ik_configs_dh.extend(approach_configs)
q_seed = approach_configs[-1].copy()

print(f"\n--- Phase 2: Solving IK for the remaining {len(TARGETS)-1} waypoints ---")
for idx in range(1, len(TARGETS)):
    target_pos = TARGETS[idx]
    q_ik = inverse_kinematics_pos(target_pos, q_seed)
    ik_configs_dh.append(q_ik.copy())
    q_seed = q_ik.copy()
    if idx % 10 == 0 or idx == len(TARGETS) - 1:
        print(f"  Solved {idx}/{len(TARGETS)-1} remaining waypoints...")

ik_configs_sim = [dh_to_sim(q) for q in ik_configs_dh]

N_STEPS     = int(TARGET_DUR / DT)
times       = [i * DT for i in range(N_STEPS)]
configs_sim = resample_to_n(ik_configs_sim, N_STEPS)
move_id     = 'waypoint_path_smoothed'


# =====================================================================
# PHASE 3: START SIMULATION → OPEN GRIPPER → MOVE → GRAB CUP
# =====================================================================
print("\n--- Phase 3: Execute trajectory and grab cup ---")

# ── 3-A  Start simulation in stepping (synchronous) mode ──────────
sim.setStepping(True)
sim.startSimulation()

if sim.getSimulationState() == sim.simulation_stopped:
    print("  ✘ Simulation failed to start.")
    sim.setStepping(False)
    exit()

# ── Wait until the yaskawa Lua script signals 'ready' ─────────────
# The script sets the signal to 'ready' once it has fully initialised.
# We keep stepping until we see it — this is the only reliable way.
print("  Waiting for robot script to initialise", end="", flush=True)
INIT_TIMEOUT = 15.0   # seconds — raise if your machine is slow
t0 = time.time()
while True:
    sim.step()
    sig = sim.getStringSignal(STR_SIGNAL)
    if isinstance(sig, bytes):
        sig = sig.decode()
    if sig == 'ready':
        break
    if time.time() - t0 > INIT_TIMEOUT:
        # Script didn't send 'ready' — it may use a different boot signal.
        # Fall back: just keep stepping for 3 more seconds and proceed anyway.
        print(f"  [WARN] 'ready' signal not received after {INIT_TIMEOUT}s.")
        print("  Falling back: stepping 60 extra ticks and continuing...")
        for _ in range(60):
            sim.step()
        break
    print(".", end="", flush=True)
    time.sleep(0.01)
print(" OK")

OPEN_VEL  =  0.04   # positive = open
CLOSE_VEL = -0.04   # negative = close

# ── 3-B  Open gripper by injecting OPEN_VEL into the main trajectory ─
# IMPORTANT: Do NOT send a separate one-point dispatch to open the gripper.
# The Lua script requires at least 2 time-points and will crash/end on a
# single-point command, making all subsequent callScriptFunction calls fail
# with "script has already ended".
#
# Instead, we bake OPEN_VEL directly into the main trajectory so the
# gripper opens as the arm starts moving — one dispatch, zero risk.
print("  Gripper will open during trajectory (baked into move).")

# ── 3-C  Dispatch the full arm trajectory (gripper opens while moving) ─
print(f"  Dispatching trajectory '{move_id}' …")
try:
    dispatch(sim, configs_sim, times, move_id, gripper_vel=OPEN_VEL)
    print("  Dispatch OK.")
except Exception as e:
    print(f"  ✘ Dispatch failed: {e}")
    sim.setStepping(False)
    sim.stopSimulation()
    exit()

# ── 3-D  Monitor distance: step sim, watch gripperEF → Cup gap ────
THRESHOLD     = 0.02   # metres — trigger grab when closer than this
grab_triggered = False
end_time       = times[-1]

print(f"  Tracking… (grab triggers at dist ≤ {THRESHOLD} m)\n")

while True:
    sim.step()                          # advance the clock by one DT tick

    sim_time = sim.getSimulationTime()

    # ── Resolve positions every step using object handles (not hard-coded IDs)
    tcp_pos = np.array(sim.getObjectPosition(gripper_ef_handle, -1))
    cup_pos = np.array(sim.getObjectPosition(cup_pose_handle, -1))  # cup_pose#0 centre
    dist    = float(np.linalg.norm(tcp_pos - cup_pos))

    print(f"\r  t={sim_time:.2f}s | dist={dist:.4f}m", end="", flush=True)

    # ── Grab trigger ──────────────────────────────────────────────
    if dist <= THRESHOLD and not grab_triggered:
        print(f"\n\n  [✔] Aligned! dist={dist:.4f} m — closing gripper.")

        # Read current joint angles in sim space
        q_grab_sim = [sim.getJointPosition(h) for h in joint_handles]

        # Send "hold current pose, close gripper"
        # Use 2 identical waypoints so the Lua script gets a valid time series
        # (single-point dispatches crash the script — minimum is 2 points)
        hold_times = [0.0, DT]
        dispatch(sim, [q_grab_sim, q_grab_sim], hold_times,
                 'close_gripper', gripper_vel=CLOSE_VEL)

        # Step ~1.5 s to let fingers close fully
        for _ in range(30):
            sim.step()
            time.sleep(0.005)

        # Attach the cup to MicoHand so it moves rigidly with the arm
        # microhand_handle already resolved at top of script
        sim.setObjectParent(cup_handle, microhand_handle, True)
        print("  [✔] Cup attached to MicoHand.")

        grab_triggered = True
        break

    # ── Safety exit if trajectory finished but grab never happened ─
    if sim_time >= end_time + 0.5 and not grab_triggered:
        print(f"\n  [!] Trajectory ended. Closest dist achieved: {dist:.4f} m")
        print("      Increase THRESHOLD or check TARGETS alignment.")
        break

# ── 3-E  Cleanup ──────────────────────────────────────────────────
sim.setStepping(False)
sim.stopSimulation()
print("\n\nPhase 3 complete. Simulation stopped.")