import fastf1
import numpy as np
import pandas as pd
from pathlib import Path

fastf1.Cache.enable_cache('cache')

# fll calendar sizes (scheduled rounds, not actual run races)
race_counts = {
    2018: 21, 2019: 21, 2020: 22,
    2021: 22, 2022: 22, 2023: 22,
    2024: 24, 2025: 24
}

def compute_long_run_avg(laps):
    try:
        if laps is None or laps.empty:
            return np.nan
        fastest = laps['LapTime'].min().total_seconds()
        long_runs = laps[(laps['LapTime'].dt.total_seconds() >= fastest + 1.5) &
                         (laps['LapTime'].dt.total_seconds() <= fastest + 4.0)]
        return long_runs['LapTime'].mean().total_seconds() if not long_runs.empty else np.nan
    except Exception:
        return np.nan

def safe_weather(session):
    try:
        if session is None:
            return "unknown"
        wd = session.weather_data
        return "rainy" if wd['Rainfall'].mean() > 0 else "dry"
    except Exception:
        return "unknown"

def safe_position(driver_row):
    try:
        if driver_row is not None and not driver_row.empty and pd.notna(driver_row['Position'].iloc[0]):
            return int(driver_row['Position'].iloc[0])
        else:
            return np.nan
    except Exception:
        return np.nan

def build_year(year):
    Path("datasets").mkdir(exist_ok=True)
    file_path = f"datasets/f1_{year}_all_drivers.csv"
    open(file_path, 'w').close()  # clear file
    
    driver_points_hist = {}
    team_points_hist = {}
    
    for rnd in range(1, race_counts[year]+1):
        race_id = f"{year%100}-{rnd:02d}"
        print(f"Processing {year} Round {rnd}...")
        
        try:
            race = fastf1.get_session(year, rnd, 'R')
            race.load()
        except Exception:
            print(f"Skipping {race_id} (race not available)")
            continue   # skip this race only
        
        # Try to get results and laps safely
        try:
            race_results = race.results
        except Exception:
            print(f"Skipping {race_id} (no results)")
            continue   # skip this race only
        try:
            race_laps = race.laps
        except Exception:
            print(f"Skipping {race_id} (no laps)")
            continue   # skip this race only
        
        track_type = race.event.get('EventName', "unknown")
        race_weather = safe_weather(race)
        drivers = getattr(race, "drivers", [])
        
        
        def load_session(stype):
            try:
                s = fastf1.get_session(year, rnd, stype)
                s.load()
                return s
            except Exception:
                return None
        
        fp1 = load_session('FP1')
        fp2 = load_session('FP2')
        fp3 = load_session('FP3')
        quali = load_session('Q')
        sprint = load_session('S')
        
        rows = []
        for drv in drivers:
            try:
                drv_info = race.get_driver(drv)
                drv_code = drv_info.get('Abbreviation', "UNK")
                team_name = drv_info.get('TeamName', "UNK")
            except Exception:
                drv_code, team_name = "UNK", "UNK"
            
            row = [race_id]
            
            # FP1
            try:
                laps = fp1.laps.pick_driver(drv_code) if fp1 else None
                fp1_avg = compute_long_run_avg(laps)
                fp1_weather = safe_weather(fp1)
            except Exception:
                fp1_avg, fp1_weather = np.nan, "unknown"
            row.extend([fp1_avg, fp1_weather])
            
            # FP2 / Sprint
            fp2_avg, fp2_weather = np.nan, np.nan
            fp3_avg, fp3_weather = np.nan, np.nan
            try:
                if race.event.get("EventFormat", "") == 'sprint':
                    laps = sprint.laps.pick_driver(drv_code) if sprint else None
                    fp3_avg = compute_long_run_avg(laps)
                    fp3_weather = safe_weather(sprint)
                else:
                    laps = fp2.laps.pick_driver(drv_code) if fp2 else None
                    fp2_avg = compute_long_run_avg(laps)
                    fp2_weather = safe_weather(fp2)
                    laps = fp3.laps.pick_driver(drv_code) if fp3 else None
                    fp3_avg = compute_long_run_avg(laps)
                    fp3_weather = safe_weather(fp3)
            except Exception:
                pass
            row.extend([fp2_avg, fp2_weather, fp3_avg, fp3_weather])
            
            # Qualifying
            try:
                res = quali.results if quali else None
                driver_row = res[res['Abbreviation'] == drv_code] if res is not None else None
                quali_pos = safe_position(driver_row)
                quali_weather = safe_weather(quali)
            except Exception:
                quali_pos, quali_weather = np.nan, "unknown"
            row.extend([quali_pos, quali_weather])
            
            # Performance indices
            d_hist = driver_points_hist.get(drv_code, [])
            t_hist = team_points_hist.get(team_name, [])
            driver_perf = sum(d_hist[-5:])/(len(d_hist[-5:])*25) if d_hist else 0.0
            team_perf = sum(t_hist[-5:])/(len(t_hist[-5:])*43) if t_hist else 0.0
            row.extend([driver_perf, team_perf])
            
            # Race pace + finishing position
            try:
                laps = race_laps.pick_driver(drv_code)
                avg_race_pace = laps['LapTime'].mean().total_seconds() if not laps.empty else np.nan
            except Exception:
                avg_race_pace = np.nan
            try:
                driver_row = race_results[race_results['Abbreviation'] == drv_code]
                finishing_pos = safe_position(driver_row)
            except Exception:
                finishing_pos = np.nan
            row.extend([track_type, race_weather, avg_race_pace, finishing_pos])
            
            # Update rolling points
            try:
                driver_row = race_results[race_results['Abbreviation'] == drv_code]
                if not driver_row.empty:
                    pts = driver_row['Points'].iloc[0]
                    driver_points_hist.setdefault(drv_code, []).append(pts)
                    team_pts = race_results[race_results['TeamName'] == team_name]['Points'].sum()
                    team_points_hist.setdefault(team_name, []).append(team_pts)
            except Exception:
                pass
            
            rows.append(row)
        
        
        df_race = pd.DataFrame(rows, columns=[
            "race_id",
            "fp1_long_run","fp1_weather",
            "fp2_long_run","fp2_weather",
            "fp3_long_run","fp3_weather",
            "qualifying","qualifying_weather",
            "driver_perf","team_perf",
            "track_type","race_weather",
            "race_pace","finishing_position"
        ])
        write_header = (rnd == 1)
        df_race.to_csv(file_path, mode='a', header=write_header, index=False)
    
    print(f"Year {year} finished → {file_path}")
    return file_path

for year in race_counts.keys():
    build_year(year)
