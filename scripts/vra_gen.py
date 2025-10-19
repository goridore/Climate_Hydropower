import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load from your saved CSV ---
df = pd.read_csv("./data/gen.csv")  # adjust path if needed

# Ensure 'year' is numeric and sorted
df["year"] = df["year"].astype(int)
df = df.sort_values("year").reset_index(drop=True)

# --- Fill missing years ---
# 1️⃣ Ensure full range (1981–2023)
full_years = pd.DataFrame({"year": np.arange(df["year"].min(), df["year"].max() + 1)})
df = pd.merge(full_years, df, on="year", how="left")

# 2️⃣ Interpolate missing generation values
df["generation"] = df["generation"].interpolate(method="linear")

# 3️⃣ Optional smoothing (for visual realism)
df["generation_smooth"] = df["generation"].rolling(window=3, center=True, min_periods=1).mean()

# --- Save the filled data ---
df.to_csv("./outputs/hydropower_filled.csv", index=False)

# --- Visualize ---
plt.figure(figsize=(10, 5))
plt.plot(df["year"], df["generation"], color="gray", lw=1.5, label="Interpolated")
plt.plot(df["year"], df["generation_smooth"], color="blue", lw=2, label="Smoothed (3-year mean)")
plt.title("Hydropower Generation (Filled & Smoothed)")
plt.ylabel("Generation (GWh)")
plt.xlabel("Year")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# Save to file
plot_path = "./outputs/hydropower_trend.png"
plt.savefig(plot_path, dpi=300)
plt.close()


print(df.head(10))
