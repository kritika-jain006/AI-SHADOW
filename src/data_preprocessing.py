import pandas as pd
import os

BASE_PATH = "data/raw"
INTERIM_PATH = "data/interim/master_dataset.csv"

# YEARS = [2020, 2021, 2022, 2023, 2024]
YEARS=[2022]

def load_year_data(year):
    year_folder = os.path.join(BASE_PATH, str(year))

    emp_path = os.path.join(year_folder, f"emp_up{str(year)[-2:]}.csv")
    financial_path = os.path.join(year_folder, f"financial_up{str(year)[-2:]}.csv")

    try:
        emp = pd.read_csv(emp_path, header=4)
        financial = pd.read_csv(financial_path, header=3)

        print("EMP COLUMNS:", emp.columns)
        print("FIN COLUMNS:", financial.columns)

    except Exception as e:
        print(f"Skipping year {year} due to file issue: {e}")
    return None

    # Clean column names
    for df in [emp, financial]:
        df.columns = df.columns.str.strip().str.lower()
        df.rename(columns=lambda x: "district" if "district" in x else x, inplace=True)

    # Validate district column
    if not all("district" in df.columns for df in [emp, financial]):
        print(f"District column missing in {year}")
        return None

    # Merge employment + financial only
    df = emp.merge(financial, on="district", how="inner")

    df["year"] = year

    return df


def build_master_dataset():
    all_data = []

    for year in YEARS:
        print(f"Processing year {year}...")
        df_year = load_year_data(year)

        if df_year is not None:
            all_data.append(df_year)

    if not all_data:
        raise ValueError("No valid yearly data found. Check raw files.")

    master_df = pd.concat(all_data, ignore_index=True)
    master_df.sort_values(by=["district", "year"], inplace=True)

    os.makedirs("data/interim", exist_ok=True)
    master_df.to_csv(INTERIM_PATH, index=False)

    print("Master dataset created successfully.")
    return master_df


if __name__ == "__main__":
    build_master_dataset()