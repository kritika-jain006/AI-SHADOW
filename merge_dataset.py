import pandas as pd

YEARS = [2020, 2021, 2022, 2023, 2024]

emp_list = []
fin_list = []

for year in YEARS:

    emp = pd.read_csv(f"data/clean/emp_{year}.csv")
    fin = pd.read_csv(f"data/clean/financial_{year}.csv")

    emp_list.append(emp)
    fin_list.append(fin)

# combine all years
emp_all = pd.concat(emp_list, ignore_index=True)
fin_all = pd.concat(fin_list, ignore_index=True)
merged = pd.merge(
    emp_all,
    fin_all,
    on=["District", "Year"],
    how="inner"
)
merged.to_csv("data/master_dataset.csv", index=False)

print("Master dataset created")
print(merged.shape)
print(merged.head())