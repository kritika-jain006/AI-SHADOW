import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/predictions/district_risk_predictions.csv")

risk_by_year = df.groupby("Year")["predicted_risk"].sum()

risk_by_year.plot(kind="bar")

plt.title("Governance Risk Events by Year")
plt.xlabel("Year")
plt.ylabel("Number of Risk Events")

plt.show()