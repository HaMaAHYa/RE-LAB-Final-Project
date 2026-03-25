import pandas as pd

# 1. Load the CSV file
df = pd.read_csv('../Assignment/conveyor/cup_trajectory_data.csv')

start_time = 4.0
end_time = 6.0

filtered_df = df[(df['Time(s)'] >= start_time) & (df['Time(s)'] <= end_time)]

# 3. Keep only the Time and XYZ columns
xyz_data = filtered_df[['X(m)', 'Y(m)', 'Z(m)']]

# 4. Save the result to a new CSV file
xyz_data.to_csv(f'Trajectory_Designer/extracted_xyz_{start_time}_to_{end_time}.csv', index=False)

# Optional: Print to verify
print(xyz_data)