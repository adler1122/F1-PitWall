import pandas as pd

df= pd.read_csv("datasets/f1_merged_all_years.csv")

tracks = df["track_type"].unique()

#for track in tracks:
    #print(track)

track_environment={
                    "street":
                        ["Azerbaijan Grand Prix","Monaco Grand Prix","Singapore Grand Prix","Saudi Arabian Grand Prix",
                         "Miami Grand Prix"],
                    "hybrid":
                        ["Australian Grand Prix","Canadian Grand Prix","Russian Grand Prix"],
                    "classic":
                        ["Bahrain Grand Prix","Chinese Grand Prix","Spanish Grand Prix","French Grand Prix",
                         "Austrian Grand Prix","British Grand Prix","German Grand Prix","Hungarian Grand Prix",
                         "Belgian Grand Prix","Italian Grand Prix","Japanese Grand Prix","United States Grand Prix",
                         "Mexican Grand Prix","Brazilian Grand Prix","Styrian Grand Prix","Abu Dhabi Grand Prix",
                         "70th Anniversary Grand Prix","Tuscan Grand Prix","Portuguese Grand Prix","Emilia Romagna Grand Prix",
                         "Turkish Grand Prix","Sakhir Grand Prix","Dutch Grand Prix","Mexico City Grand Prix","Qatar Grand Prix"]       
                }

setup_demand={
              "high down force":
                ["Spanish Grand Prix","Monaco Grand Prix","Hungarian Grand Prix","Singapore Grand Prix",
                 "Japanese Grand Prix","Mexican Grand Prix","Tuscan Grand Prix","Emilia Romagna Grand Prix",
                 "Turkish Grand Prix","Dutch Grand Prix","Mexico City Grand Prix","Qatar Grand Prix"],
              "medium down force":
                ["Australian Grand Prix","Bahrain Grand Prix","Chinese Grand Prix","Russian Grand Prix",
                 "Austrian Grand Prix","British Grand Prix","German Grand Prix","Belgian Grand Prix",
                 "United States Grand Prix","Brazilian Grand Prix","Styrian Grand Prix","Abu Dhabi Grand Prix",
                 "70th Anniversary Grand Prix","Portuguese Grand Prix","Saudi Arabian Grand Prix",
                 "Miami Grand Prix"],
              "low down force":
                ["Azerbaijan Grand Prix","Canadian Grand Prix","French Grand Prix","Italian Grand Prix",
                 "Sakhir Grand Prix"]
                }


track_length={"Australian Grand Prix":5.278
              ,"Bahrain Grand Prix":5.412
              ,"Chinese Grand Prix":5.451
              ,"Azerbaijan Grand Prix":6.003
              ,"Spanish Grand Prix":4.657
              ,"Monaco Grand Prix":3.337
              ,"Canadian Grand Prix":4.361
              ,"French Grand Prix":5.842
              ,"Austrian Grand Prix":4.318
              ,"British Grand Prix":5.891
              ,"German Grand Prix":4.574
              ,"Hungarian Grand Prix":4.381
              ,"Belgian Grand Prix":7.004
              ,"Italian Grand Prix":5.793
              ,"Singapore Grand Prix":5.063
              ,"Russian Grand Prix":5.848
              ,"Japanese Grand Prix":5.807
              ,"United States Grand Prix":5.513
              ,"Mexican Grand Prix":4.304
              ,"Brazilian Grand Prix":4.309
              ,"Styrian Grand Prix":4.318
              ,"Abu Dhabi Grand Prix":5.281
              ,"70th Anniversary Grand Prix":5.819
              ,"Tuscan Grand Prix":5.245
              ,"Portuguese Grand Prix":4.653
              ,"Emilia Romagna Grand Prix":4.909
              ,"Turkish Grand Prix":5.338
              ,"Sakhir Grand Prix":3.543
              ,"Dutch Grand Prix":4.259
              ,"Mexico City Grand Prix":4.304
              ,"Saudi Arabian Grand Prix":6.174
              ,"Miami Grand Prix":5.412
              ,"Qatar Grand Prix":5.419
              }

#for track in tracks:
    #if track not in track_length.keys():
        #print(track)


df["track_environment"] = df["track_type"].map(
    {gp: env for env, gps in track_environment.items() for gp in gps})

df["setup_demand"] = df["track_type"].map(
    {gp: setup for setup, gps in setup_demand.items() for gp in gps})

df["track_length"] = df["track_type"].map(track_length)

df = df.drop(columns=["track_type"])


df["qualifying"] = df["qualifying"].astype("Int64")
df["finishing_position"] = df["finishing_position"].astype("Int64")


df["grid_size"] = df.groupby("race_id")["qualifying"].transform("count")


df["relative_qualifying"] = df["qualifying"] / df["grid_size"]
df["relative_finish"] = df["finishing_position"] / df["grid_size"]


df = df.drop(columns=["qualifying", "finishing_position"])



# encode weather columns: dry=0, rainy=1
weather_map = {"dry": 0, "rainy": 1}
weather_cols = ["fp1_weather","fp2_weather","fp3_weather","qualifying_weather","race_weather"]
df[weather_cols] = df[weather_cols].map(lambda x: weather_map.get(x, x))

# encode setup_demand: high down force=0, medium down force=1, low down force=2
setup_map = {"high down force": 0, "medium down force": 1, "low down force": 2}
df["setup_demand"] = df["setup_demand"].map(setup_map)
df["setup_demand"] = df["setup_demand"].astype("Int64")
# encode track_environment: street=0, hybrid=1, classic=2
env_map = {"street": 0, "hybrid": 1, "classic": 2}
df["track_environment"] = df["track_environment"].map(env_map)
df["track_environment"] = df["track_environment"].astype("Int64")
#print(df.head().to_string()) 

#print(df.dtypes)


df = df.drop(columns=["grid_size"])


col_order = [
    "race_id",
    "fp1_weather",
    "fp2_weather",
    "fp3_weather",
    "qualifying_weather",
    "race_weather",
    "track_environment",
    "setup_demand",       
    "track_length",
    "driver_perf",
    "team_perf",
    "fp1_long_run",
    "fp2_long_run",
    "fp3_long_run",
    "race_pace",
    "relative_qualifying",
    "relative_finish",    
]

df = df[col_order]

df.to_csv("f1_encoded_not_scaled.csv", index=False)
