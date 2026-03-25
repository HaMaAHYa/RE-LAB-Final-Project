import pandas as pd
import numpy as np
import math

# 1. Load the CSV file
df = pd.read_csv('../Assignment/conveyor/cup_trajectory_data.csv')

start_time = 2.0
end_time = 10.0

# --- NEW: S-Curve Z-Increase Parameters ---
increase_start_time = 5.0  # The time 'x' when Z should start increasing
increase_duration = 4.0    # How long the smooth lift takes (seconds)
total_z_increase = 0.4     # Total distance to lift in Z (meters)
# ------------------------------------------

# 2. Filter data by time
# Note: We use .copy() here to safely modify the dataframe without getting Pandas warnings
filtered_df = df[(df['Time(s)'] >= start_time) & (df['Time(s)'] <= end_time)].copy()

# Function to calculate smooth S-curve offset
def calculate_s_curve_offset(t_current, t_start, duration, total_dist):
    if t_current <= t_start:
        return 0.0
    elif t_current >= t_start + duration:
        return total_dist
    else:
        # Normalized time 't' between 0.0 and 1.0
        t = (t_current - t_start) / duration
        # 5th-order polynomial for smooth acceleration and deceleration
        s = 6 * (t**5) - 15 * (t**4) + 10 * (t**3)
        return s * total_dist

# Apply the S-curve offset to the existing Z values
filtered_df['Z_offset'] = filtered_df['Time(s)'].apply(
    lambda t: calculate_s_curve_offset(t, increase_start_time, increase_duration, total_z_increase)
)
filtered_df['Z(m)'] += filtered_df['Z_offset']

# 3. Format each row
formatted_lines = []
for index, row in filtered_df.iterrows():
    # Rounding to 6 decimal places keeps the output file clean
    x = round(row['X(m)'], 6)
    y = round(row['Y(m)'], 6)
    z = round(row['Z(m)'], 6)
    
    # Formats them into the 3D string required for the robot IK script
    line = f"np.array([{x}, {y}, {z}]),"
    formatted_lines.append(line)

# 4. Save the result to a text file
output_path = f'Trajectory_Designer/extracted_xyz_{start_time}_to_{end_time}.csv'

with open(output_path, 'w') as f:
    for item in formatted_lines:
        f.write("%s\n" % item)

# Optional: Print out a quick sample to verify
print(f"Saved {len(formatted_lines)} positions to {output_path}")
for line in formatted_lines[:5]:
    print(line)