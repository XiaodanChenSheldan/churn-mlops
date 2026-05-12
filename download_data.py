# download_data.py
from pathlib import Path

import pandas as pd

# Create data directory
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Download Telco Customer Churn dataset (IBM's public dataset)
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Save locally
df.to_csv("data/raw/churn.csv", index=False)

print(f"Downloaded {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
