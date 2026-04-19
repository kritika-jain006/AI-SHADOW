
import pandas as pd

def clean_emp(path):

    df = pd.read_csv(path, header=None)

    # skip top rows containing titles
    df = df.iloc[3:].reset_index(drop=True)

    # manually set column names
    columns = ["SNo", "District"]

    for i in range(len(df.columns) - 2):
        columns.append(f"metric_{i+1}")

    df.columns = columns

    return df

def clean_financial(path):

    df = pd.read_csv(path, header=None)

    # skip title rows
    df = df.iloc[13:].reset_index(drop=True)

    columns = ["SNo", "District"]

    for i in range(len(df.columns) - 2):
        columns.append(f"financial_metric_{i+1}")

    df.columns = columns

    return df

import os

BASE_PATH = "data/raw"
SAVE_PATH = "data/clean"

YEARS = [2020, 2021, 2022, 2023, 2024]

for year in YEARS:

    year_path = os.path.join(BASE_PATH, str(year))

    emp_file = os.path.join(year_path, f"emp_up{str(year)[-2:]}.csv")
    fin_file = os.path.join(year_path, f"financial_up{str(year)[-2:]}.csv")

    emp_df = clean_emp(emp_file)
    fin_df = clean_financial(fin_file)

    emp_df["Year"] = year
    fin_df["Year"] = year

    emp_df.to_csv(f"{SAVE_PATH}/emp_{year}.csv", index=False)
    fin_df.to_csv(f"{SAVE_PATH}/financial_{year}.csv", index=False)

print("All datasets cleaned successfully")