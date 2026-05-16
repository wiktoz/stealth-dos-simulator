import pandas as pd
from pathlib import Path
from zat.log_to_dataframe import LogToDataFrame

log_to_df = LogToDataFrame()
base_dir = Path("../zeek_out")

def load_log(folder_path, filename):
    path = folder_path / filename

    if not path.exists():
        return None

    try:
        df = log_to_df.create_dataframe(str(path))
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def normalize_ts(df):

    # ts may be index OR column depending on zat behavior
    if df.index.name == "ts":
        df = df.reset_index()

    if "ts" not in df.columns:
        return None

    # ensure numeric timestamp
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    return df

def process_subfolder(folder_path):

    print(f"Processing: {folder_path.name}")

    conn = load_log(folder_path, "conn.log")

    if conn is None:
        return None

    conn = normalize_ts(conn)

    if conn is None:
        print("Missing ts in conn.log after normalization")
        return None

    dfs = [conn]

    # optional enrichment logs
    # enrichment_logs = ["http.log", "dns.log", "ssl.log"]
    enrichment_logs = ["http.log"]
    for log_name in enrichment_logs:

        df = load_log(folder_path, log_name)

        if df is None:
            continue

        df = normalize_ts(df)

        if df is not None:
            dfs.append(df)


    combined = pd.concat(dfs, ignore_index=True)

    # remove invalid timestamps
    if "ts" in combined.columns:
        combined = combined.dropna(subset=["ts"])
        combined = combined.sort_values("ts")

    return combined


all_dfs = []

for subfolder in base_dir.iterdir():

    if subfolder.is_dir():

        df = process_subfolder(subfolder)

        if df is not None:
            all_dfs.append(df)


if all_dfs:

    final_dataset = pd.concat(all_dfs, ignore_index=True)

    if "ts" in final_dataset.columns:
        final_dataset = final_dataset.sort_values("ts")

    final_dataset = final_dataset.reset_index(drop=True)

    final_dataset.to_csv("dataset.csv", index=False)

    print("\nDONE")
    print("Shape:", final_dataset.shape)
    print("Columns:", list(final_dataset.columns))

else:
    print("No data found")