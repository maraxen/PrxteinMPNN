import pandas as pd
import numpy as np

df = pd.read_csv("comparison_validity.csv")

print("Overall Statistics:")
print(df.groupby("engine")[["validity", "phi_mean", "psi_mean"]].agg(["mean", "std"]))

print("\nProduction Phase Statistics:")
prod_df = df[df["phase"] == "Prod"]
print(prod_df.groupby("engine")[["validity", "phi_mean", "psi_mean"]].agg(["mean", "std"]))

# Check for divergence
# Calculate rolling mean validity
df["validity_rolling"] = df.groupby("engine")["validity"].transform(lambda x: x.rolling(10, min_periods=1).mean())

print("\nFirst 10 steps:")
print(df.head(20))
