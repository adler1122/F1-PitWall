import pandas as pd
import numpy as np
import glob


# merge all datasets
files = sorted(glob.glob("datasets/f1_*_all_drivers.csv"))
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
print("merged shape:", df.shape)


# inspect missing fp long runs
fp_cols = ["fp1_long_run", "fp2_long_run", "fp3_long_run"]
print("\nInitial missing counts:")
print(df[fp_cols].isna().sum())
print("Rows with all three missing:", df[fp_cols].isna().all(axis=1).sum())


# fill missing values from same row (driver’s own other FP sessions)
for i, row in df.iterrows():
    vals = [row["fp1_long_run"], row["fp2_long_run"], row["fp3_long_run"]]
    known = [v for v in vals if pd.notna(v)]
    if len(known) >= 1:
        mean_val = np.mean(known)
        for j, col in enumerate(fp_cols):
            if pd.isna(vals[j]):
                df.at[i, col] = mean_val

print(df[fp_cols].isna().sum())
print("rows with all three missing:", df[fp_cols].isna().all(axis=1).sum())


# fill rows with all three missing using teammate rows 
# only consider rows with same team performance and same race_id
still_missing = df[df[fp_cols].isna().all(axis=1)]
print("\nRows still empty after same-row fill:", len(still_missing))

for i, row in still_missing.iterrows():
    race_id = row["race_id"]
    team_perf = row["team_perf"]

    # look at same race and same team performance
    candidates = df[(df["race_id"] == race_id) &
                    (df["team_perf"] == team_perf) &
                    (~df[fp_cols].isna().all(axis=1))]

    if len(candidates) > 0:
        ref_vals = candidates[fp_cols].values.flatten()
        ref_vals = [v for v in ref_vals if pd.notna(v)]
        if ref_vals:
            fill_val = np.mean(ref_vals)
            for col in fp_cols:
                df.at[i, col] = fill_val

print(df[fp_cols].isna().sum())
print("rows with all three missing:", df[fp_cols].isna().all(axis=1).sum())


# fill remaining rows with track/weather/team performance similarity
still_missing = df[df[fp_cols].isna().all(axis=1)]
print(len(still_missing))

for i, row in still_missing.iterrows():
    race_id = row["race_id"]
    track_type = row["track_type"]
    team_perf = row["team_perf"]

    # fp1 fill
    fp1_weather = row["fp1_weather"]
    candidates_fp1 = df[(df["track_type"] == track_type) &
                        (df["fp1_weather"] == fp1_weather) &
                        (~df["fp1_long_run"].isna())]

    if len(candidates_fp1) > 0:
        candidates_fp1["perf_diff"] = (candidates_fp1["team_perf"] - team_perf).abs()
        nearest_fp1 = candidates_fp1.sort_values("perf_diff").head(5)
        fill_val_fp1 = nearest_fp1["fp1_long_run"].mean()
        df.at[i, "fp1_long_run"] = fill_val_fp1

    # fp2 fill
    fp2_weather = row["fp2_weather"]
    candidates_fp2 = df[(df["track_type"] == track_type) &
                        (df["fp2_weather"] == fp2_weather) &
                        (~df["fp2_long_run"].isna())]

    if len(candidates_fp2) > 0:
        candidates_fp2["perf_diff"] = (candidates_fp2["team_perf"] - team_perf).abs()
        nearest_fp2 = candidates_fp2.sort_values("perf_diff").head(5)
        fill_val_fp2 = nearest_fp2["fp2_long_run"].mean()
        df.at[i, "fp2_long_run"] = fill_val_fp2

    # fp3 fill
    fp3_weather = row["fp3_weather"]
    candidates_fp3 = df[(df["track_type"] == track_type) &
                        (df["fp3_weather"] == fp3_weather) &
                        (~df["fp3_long_run"].isna())]

    if len(candidates_fp3) > 0:
        candidates_fp3["perf_diff"] = (candidates_fp3["team_perf"] - team_perf).abs()
        nearest_fp3 = candidates_fp3.sort_values("perf_diff").head(5)
        fill_val_fp3 = nearest_fp3["fp3_long_run"].mean()
        df.at[i, "fp3_long_run"] = fill_val_fp3

print(df[fp_cols].isna().sum())
print("rows with all three missing:", df[fp_cols].isna().all(axis=1).sum())


# different race_id still have missing values
missing_races = df.loc[df[fp_cols].isna().all(axis=1), "race_id"].unique()
print(" races still with missing fp sessions:", missing_races)

# find races with all fp sessions missing and drop them
empty_races = df.groupby("race_id")[fp_cols].apply(lambda g: g.isna().all(axis=1).all())
empty_races = empty_races[empty_races].index.tolist()

print(empty_races)
df = df[~df["race_id"].isin(empty_races)].copy()
print(df.shape)
print(df[fp_cols].isna().sum())
print(df[fp_cols].isna().all(axis=1).sum())


# fill weather columns
weather_cols = ["fp1_weather","fp2_weather","fp3_weather","qualifying_weather","race_weather"]

# Replace "unknown" with np.nan
df[weather_cols] = df[weather_cols].replace("unknown", np.nan)

# NaN counts
print(df[weather_cols].isna().sum())

# fp1: fill NaN with fp2 weather
mask_fp1 = df["fp1_weather"].isna()
df.loc[mask_fp1, "fp1_weather"] = df.loc[mask_fp1, "fp2_weather"]

# fp2: fill NaN with fp1 weather
mask_fp2 = df["fp2_weather"].isna()
df.loc[mask_fp2, "fp2_weather"] = df.loc[mask_fp2, "fp1_weather"]

# fp3: fill NaN with qualifying weather
mask_fp3 = df["fp3_weather"].isna()
df.loc[mask_fp3, "fp3_weather"] = df.loc[mask_fp3, "qualifying_weather"]

# qualifying :fill NaN with fp3 weather
mask_quali = df["qualifying_weather"].isna()
df.loc[mask_quali, "qualifying_weather"] = df.loc[mask_quali, "fp3_weather"]

# race :fill NaN with qualifying weather
mask_race = df["race_weather"].isna()
df.loc[mask_race, "race_weather"] = df.loc[mask_race, "qualifying_weather"]

print(df[weather_cols].isna().sum())

# check that weather values are all strings ("rainy" or "dry")
for col in weather_cols:
    print(f"\nColumn: {col}")
    print("Unique values:", df[col].unique())
    print("Data types:", df[col].apply(type).unique())
    print("Count of NaNs:", df[col].isna().sum())

# fill blanks in fp3 and qualifying weather with race weather
mask_fp3 = df["fp3_weather"].isna()
df.loc[mask_fp3, "fp3_weather"] = df.loc[mask_fp3, "race_weather"]

mask_quali = df["qualifying_weather"].isna()
df.loc[mask_quali, "qualifying_weather"] = df.loc[mask_quali, "race_weather"]
# check again
for col in weather_cols:
    print(f"\nColumn: {col}")
    print("Unique values:", df[col].unique())
    print("Data types:", df[col].apply(type).unique())
    print("Count of NaNs:", df[col].isna().sum())


# inspect race pace, qualifying, and race results
cols_to_check = ["race_pace", "qualifying", "finishing_position"]

print(df[cols_to_check].isna().sum())
print( df[cols_to_check].isna().all(axis=1).sum())

# finishing_position fill
missing_finish = df[df["finishing_position"].isna()]
race_ids_missing = missing_finish["race_id"].unique()

print(race_ids_missing)

# for each race, find which position is missing and fill it
for rid in race_ids_missing:
    race_group = df[df["race_id"] == rid]
    
    # known positions in this race
    known_positions = race_group["finishing_position"].dropna().astype(int).tolist()
    
    # expected positions = 1..n 
    expected_positions = list(range(1, len(race_group) + 1))
    
    # find which positions are missing
    missing_positions = set(expected_positions) - set(known_positions)
    
    print(f"Race {rid}: missing positions {missing_positions}")
    
    # fill NaN with the missing position
    for pos in missing_positions:
        idx = race_group[race_group["finishing_position"].isna()].index
        if len(idx) > 0:
            df.at[idx[0], "finishing_position"] = pos

print(df["finishing_position"].isna().sum())

# check race IDs with missing qualifying values
missing_quali = df[df["qualifying"].isna()]
race_ids_quali_missing = missing_quali["race_id"].unique()

print(race_ids_quali_missing)
print(len(race_ids_quali_missing))

# fill qualifying for specific race IDs

# race 24-17 and 23-19 -> all qualifying missing, so copy finishing_position
for rid in ["24-17", "23-19"]:
    mask = df["race_id"] == rid
    df.loc[mask, "qualifying"] = df.loc[mask, "finishing_position"]

# race 21-05 -> one qualifying missing, fill like finishing_position
race_group = df[df["race_id"] == "21-05"]
known_positions = race_group["qualifying"].dropna().astype(int).tolist()
expected_positions = list(range(1, len(race_group) + 1))
missing_positions = set(expected_positions) - set(known_positions)

for pos in missing_positions:
    idx = race_group[race_group["qualifying"].isna()].index
    if len(idx) > 0:
        df.at[idx[0], "qualifying"] = pos

# race 19-03 -> two rows missing, fill first with 19 and second with 20
race_group = df[df["race_id"] == "19-03"]
missing_idx = race_group[race_group["qualifying"].isna()].index.tolist()
if len(missing_idx) >= 2:
    df.at[missing_idx[0], "qualifying"] = 19
    df.at[missing_idx[1], "qualifying"] = 20

print(df[cols_to_check].isna().sum())
# inspect race pace missing values
missing_race_pace = df[df["race_pace"].isna()]
race_ids_race_pace_missing = missing_race_pace["race_id"].unique()

print(len(race_ids_race_pace_missing))
print(len(race_ids_race_pace_missing))

# show how many missing rows per race
missing_counts_per_race = missing_race_pace.groupby("race_id").size()

print(missing_counts_per_race)
# remove rows with missing race pace
df = df[~df["race_pace"].isna()].copy()

print(df["race_pace"].isna().sum())

# fix finishing_position continuity per race
for rid, race_group in df.groupby("race_id"):
    positions = race_group["finishing_position"].astype(int).tolist()
    expected = list(range(1, len(race_group) + 1))
    
    # If positions are not continuous, shift them up
    if set(positions) != set(expected):
        
        # sort by finishing_position
        race_group_sorted = race_group.sort_values("finishing_position").reset_index()
        
        # reassign positions from 1..n
        new_positions = list(range(1, len(race_group_sorted) + 1))
        for idx, pos in zip(race_group_sorted["index"], new_positions):
            df.at[idx, "finishing_position"] = pos

for rid, race_group in df.groupby("race_id"):
    positions = race_group["finishing_position"].astype(int).tolist()
    expected = list(range(1, len(race_group) + 1))
    if set(positions) != set(expected):
        print(f"{rid} {sorted(positions)} vs expected {expected}")


# check for missing values
print(df.isna().sum())
rows_with_missing = df.isna().any(axis=1).sum()
print(rows_with_missing)
if rows_with_missing > 0:
    missing_race_ids = df.loc[df.isna().any(axis=1), "race_id"].unique()
    print(missing_race_ids)



# save to csv
df.to_csv("datasets/f1_all_years_preprocessed.csv", index=False)
