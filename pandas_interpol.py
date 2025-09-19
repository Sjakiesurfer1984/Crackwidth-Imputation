import pandas as pd

# Load your original CSV
df = pd.read_csv("sensor_values.csv")

# Parse datetime column (assuming first column is the timestamp)
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors='coerce')
df = df.dropna(subset=[df.columns[0]]).sort_values(df.columns[0]).set_index(df.columns[0])

# Define your gap boundaries
start_ts = pd.to_datetime("2025-07-07 12:34", dayfirst=True)
end_ts   = pd.to_datetime("2025-09-01 14:49", dayfirst=True)

# Determine sampling frequency from the data (seconds)
diffs = df.index.to_series().diff().dropna().dt.total_seconds()
freq_seconds = int(diffs.mode().iloc[0]) if not diffs.mode().empty else int(diffs.median())
target_freq = f"{freq_seconds}S"

# Build the full target index for the gap
target_index = pd.date_range(start=start_ts, end=end_ts, freq=target_freq)

# Reindex region and interpolate numeric columns
region = df.loc[start_ts:end_ts]
region_interp = region.reindex(target_index)

numeric_cols = df.select_dtypes("number").columns
region_interp[numeric_cols] = region_interp[numeric_cols].interpolate(method="time", limit_direction="both")

# Replace original gap with interpolated data
df_updated = df.drop(df.loc[start_ts:end_ts].index, errors="ignore")
df_updated = pd.concat([df_updated, region_interp]).sort_index()

# Save to CSV
df_updated.to_csv("sensor_values_interpolated.csv")

print("✅ Interpolated file saved as 'sensor_values_interpolated.csv'")
