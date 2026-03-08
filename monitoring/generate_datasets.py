import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

print("Loading dataset...")

data = fetch_california_housing(as_frame=True)
df = data.frame

# Split reference / current
reference, current = train_test_split(df, test_size=0.3, random_state=42)

# Drift Simulation
current_drifted = current.copy()
current_drifted["MedInc"] *= np.random.uniform(1.3, 1.8, size=len(current_drifted))

# Save
reference.to_csv("data/reference.csv", index=False)
current_drifted.to_csv("data/current.csv", index=False)

print("Datasets generated with simulated drift")