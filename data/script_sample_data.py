import pandas as pd
import numpy as np

# Parameters
num_days = 100
start_date = "2025-01-01"

# Generate datetime column
dates = pd.date_range(start=start_date, periods=num_days, freq='D')

# Generate synthetic data
np.random.seed(42)
sales = np.random.randint(50, 200, size=num_days).cumsum()  # increasing trend
temperature = np.random.normal(20, 5, size=num_days)        # daily temperature
website_visits = np.random.randint(100, 1000, size=num_days)

# Create DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Sales": sales,
    "Temperature": temperature,
    "Website_Visits": website_visits
})

# Optional: introduce some missing values
for col in ["Sales", "Temperature"]:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

# Save to CSV
df.to_csv("data/sample_timeseries.csv", index=False)
print("Sample dataset saved as data/sample_timeseries.csv")
