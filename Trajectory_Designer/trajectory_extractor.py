import pandas as pd
import numpy as np

# 1. Load the CSV file
df = pd.read_csv('../Assignment/conveyor/cup_trajectory_data.csv')

start_time = 0.5
end_time = 10.0

# 2. Filter data by time
filtered_df = df[(df['Time(s)'] >= start_time) & (df['Time(s)'] <= end_time)]

# 3. Format each row as the string "np.array([x, y, z])"
formatted_lines = []
for index, row in filtered_df.iterrows():
    # Extracts x, y, z and formats them into the string
    line = f"np.array([{row['X(m)']}, {row['Y(m)']}, {row['Z(m)']}]),"
    formatted_lines.append(line)

# 4. Save the result to a text file
output_path = f'Trajectory_Designer/extracted_xyz_{start_time}_to_{end_time}.csv'

with open(output_path, 'w') as f:
    for item in formatted_lines:
        f.write("%s\n" % item)

# Optional: Print the first 5 to verify
print(f"Saved {len(formatted_lines)} positions to {output_path}")
for line in formatted_lines[:5]:
    print(line)