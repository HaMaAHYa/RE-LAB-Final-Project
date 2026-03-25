"""
Yaskawa DH Frame Visualiser
============================
Reads the classical DH table (derived from the URDF) and plots all 7 coordinate
frames (base + 6 joint frames) in a single 3-D figure.

Each frame is drawn as three coloured arrows:
    red   → x-axis
    green → y-axis
    blue  → z-axis

The arm skeleton (line through all frame origins) is shown in grey.
Joint labels are placed at each origin.

Run:
    python plot_frames.py
or optionally pass joint angles (deg) on the command line:
    python plot_frames.py 0 30 60 0 45 0
"""

import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

Pi = math.pi

# ── Arrow scale (fraction of total arm reach) ──────────────────────────────
ARROW_SCALE = 0.08   # metres — tweak if arrows look too big / small


# =============================================================================
# 1.  Classical DH table  (from the URDF conversation)
# =============================================================================
#
#  Joint  | theta_offset | d (m)  | a (m)  | alpha (rad)
#  -------+--------------+--------+--------+------------
#    1    |     0        | 0.330  | 0.040  |   +π/2
#    2    |     0        | 0      | 0.345  |    0
#    3    |    +π/2      | 0      | 0.040  |   +π/2
#    4    |     0        | 0.340  | 0      |   −π/2
#    5    |     0        | 0      | 0      |   +π/2
#    6    |     0        | 0.080  | 0      |    0
# =============================================================================

DH_TABLE = [
    # joint  theta_offset   d        a      alpha
    {'joint': 1, 'theta': 0.0,   'd': 0.330, 'a': 0.040, 'alpha':  Pi/2},
    {'joint': 2, 'theta': 0.0,   'd': 0.0,   'a': 0.345, 'alpha':  0.0 },
    {'joint': 3, 'theta': Pi/2,  'd': 0.0,   'a': 0.040, 'alpha':  Pi/2},
    {'joint': 4, 'theta': 0.0,   'd': 0.340, 'a': 0.0,   'alpha': -Pi/2},
    {'joint': 5, 'theta': 0.0,   'd': 0.0,   'a': 0.0,   'alpha':  Pi/2},
    {'joint': 6, 'theta': 0.0,   'd': 0.080, 'a': 0.0,   'alpha':  0.0 },
]

# Joint limits from URDF  [lower, upper]  in radians
JOINT_LIMITS = [
    (-2.967,  2.967),
    (-1.134,  2.531),
    (-1.222,  3.316),
    (-3.316,  3.316),
    (-2.356,  2.356),
    (-6.294,  6.294),
]


# =============================================================================
# 2.  Kinematics helpers
# =============================================================================

def dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Classical DH 4×4 transform (closed-form, no intermediate matrices).

        T = Rz(θ) · Tz(d) · Tx(a) · Rx(α)

        ┌  cθ   -sθ·cα    sθ·sα    a·cθ ┐
        │  sθ    cθ·cα   -cθ·sα    a·sθ │
        │   0    sα        cα       d   │
        └   0    0         0        1   ┘
    """
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ ct, -st*ca,  st*sa,  a*ct ],
        [ st,  ct*ca, -ct*sa,  a*st ],
        [  0,     sa,     ca,     d ],
        [  0,      0,      0,     1 ],
    ], dtype=float)


def all_frames(q: list) -> list:
    """
    Return a list of 7 frames:
        frames[0] = world/base frame (4×4 identity)
        frames[i] = T_0^i  for i = 1..6
    """
    frames = [np.eye(4)]
    T = np.eye(4)
    for row, qi in zip(DH_TABLE, q):
        theta = row['theta'] + qi
        T = T @ dh_matrix(theta, row['d'], row['a'], row['alpha'])
        frames.append(T.copy())
    return frames


# =============================================================================
# 3.  Drawing helpers
# =============================================================================

def draw_frame(ax, T: np.ndarray, scale: float, label: str, alpha: float = 1.0):
    """
    Draw one coordinate frame as three coloured quivers.
    T  : 4×4 homogeneous transform (origin + rotation)
    """
    origin = T[:3, 3]
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]

    kw = dict(length=scale, normalize=True, alpha=alpha,
              linewidth=2, arrow_length_ratio=0.25)

    ax.quiver(*origin, *x_axis, color='red',   **kw)
    ax.quiver(*origin, *y_axis, color='green',  **kw)
    ax.quiver(*origin, *z_axis, color='blue',   **kw)

    # Label slightly above the origin
    offset = z_axis * scale * 0.5
    ax.text(*(origin + offset), label,
            fontsize=9, fontweight='bold',
            color='black', ha='center', va='bottom')


def draw_skeleton(ax, frames: list):
    """
    Draw grey lines connecting consecutive frame origins (the arm skeleton).
    """
    origins = np.array([T[:3, 3] for T in frames])
    ax.plot(origins[:, 0], origins[:, 1], origins[:, 2],
            'o--', color='gray', linewidth=1.2, markersize=5,
            markerfacecolor='white', markeredgecolor='gray', zorder=1)


def set_equal_axes(ax, frames: list, margin: float = 0.1):
    """
    Force equal aspect ratio on all three axes (matplotlib 3-D workaround).
    """
    origins = np.array([T[:3, 3] for T in frames])
    x, y, z = origins[:, 0], origins[:, 1], origins[:, 2]
    mid  = np.array([(x.max()+x.min())/2,
                     (y.max()+y.min())/2,
                     (z.max()+z.min())/2])
    rng  = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min()) / 2 + margin
    ax.set_xlim(mid[0]-rng, mid[0]+rng)
    ax.set_ylim(mid[1]-rng, mid[1]+rng)
    ax.set_zlim(0,           mid[2]+rng*1.2)


# =============================================================================
# 4.  Main plot function
# =============================================================================

def plot_robot_frames(q: list, title_suffix: str = ''):
    """
    Plot all DH frames for the given joint configuration.

    Args:
        q             : 6-element list of joint angles [rad]
        title_suffix  : appended to the window title
    """
    frames = all_frames(q)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # ── skeleton first (drawn behind the frames) ──────────────────────
    draw_skeleton(ax, frames)

    # ── coordinate frames ─────────────────────────────────────────────
    labels = ['F0\n(base)', 'F1\n(J1)', 'F2\n(J2)', 'F3\n(J3)',
              'F4\n(J4)',   'F5\n(J5)', 'F6\n(EE)']
    colors_alpha = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    for T, lbl, a in zip(frames, labels, colors_alpha):
        draw_frame(ax, T, scale=ARROW_SCALE, label=lbl, alpha=a)

    # ── legend patch ──────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='red',   linewidth=2, label='x-axis'),
        Line2D([0], [0], color='green', linewidth=2, label='y-axis'),
        Line2D([0], [0], color='blue',  linewidth=2, label='z-axis'),
        Line2D([0], [0], color='gray',  linewidth=1.5, linestyle='--', label='skeleton'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8)

    # ── axis formatting ───────────────────────────────────────────────
    set_equal_axes(ax, frames)
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_zlabel('Z (m)', fontsize=9)

    q_deg = [round(math.degrees(qi), 1) for qi in q]
    title = f'Yaskawa DH Frames{title_suffix}\nq = {q_deg} °'
    ax.set_title(title, fontsize=10)

    # Print EE position to console
    ee = frames[-1]
    print(f'\nEnd-effector position (world frame):')
    print(f'  x = {ee[0,3]:.4f} m')
    print(f'  y = {ee[1,3]:.4f} m')
    print(f'  z = {ee[2,3]:.4f} m')
    print(f'  Rotation matrix R =')
    print(ee[:3, :3])

    plt.tight_layout()
    return fig, ax


# =============================================================================
# 5.  Multi-panel: home + a sample configuration
# =============================================================================

def plot_comparison(q_home: list, q_target: list):
    """
    Side-by-side plot: home pose (left) vs target pose (right).
    """
    fig = plt.figure(figsize=(16, 7))

    for col, (q, title) in enumerate([(q_home, 'Home pose  (all zeros)'),
                                       (q_target, 'Target configuration')]):
        frames = all_frames(q)
        ax = fig.add_subplot(1, 2, col+1, projection='3d')

        draw_skeleton(ax, frames)
        labels = ['F0\n(base)', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6\n(EE)']
        for T, lbl in zip(frames, labels):
            draw_frame(ax, T, scale=ARROW_SCALE, label=lbl)

        set_equal_axes(ax, frames)
        ax.set_xlabel('X (m)', fontsize=8);  ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_zlabel('Z (m)', fontsize=8)
        q_deg = [round(math.degrees(qi), 1) for qi in q]
        ax.set_title(f'{title}\nq = {q_deg} °', fontsize=9)

    plt.suptitle('Yaskawa — DH Frame Visualisation', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# 6.  Entry point
# =============================================================================

if __name__ == '__main__':
    # Parse optional command-line joint angles (degrees)
    if len(sys.argv) == 7:
        q_input = [math.radians(float(v)) for v in sys.argv[1:]]
        # Clamp to joint limits
        q_clamped = []
        for i, (qi, (lo, hi)) in enumerate(zip(q_input, JOINT_LIMITS)):
            if qi < lo or qi > hi:
                print(f'Warning: joint {i+1} = {math.degrees(qi):.1f}° '
                      f'exceeds limits [{math.degrees(lo):.1f}, {math.degrees(hi):.1f}]. Clamping.')
            q_clamped.append(float(np.clip(qi, lo, hi)))
        q_target = q_clamped
        plot_robot_frames(q_target, title_suffix=' — custom config')
        plt.show()
    else:
        # Default: show home pose + an interesting sample configuration
        q_home   = [0.0] * 6
        q_sample = [
            math.radians( 30),   # J1  swing
            math.radians( 45),   # J2  lower arm up
            math.radians( 60),   # J3  upper arm
            math.radians(  0),   # J4  wrist roll
            math.radians( 30),   # J5  wrist bend
            math.radians(  0),   # J6  wrist turn
        ]

        print('=== HOME POSE ===')
        plot_robot_frames(q_home, title_suffix=' — home pose')

        print('\n=== SAMPLE CONFIG (30, 45, 60, 0, 30, 0) deg ===')
        plot_comparison(q_home, q_sample)

        plt.show()