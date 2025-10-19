import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("📥 Loading datasets...")
climate = pd.read_csv("./outputs/volta_annual_climate.csv")
hydro = pd.read_csv("./outputs/hydropower_filled.csv")

# Merge by year
df = pd.merge(climate, hydro, on="year", how="inner")

# --- 1️⃣ Estimate Potential Evapotranspiration (PET) ---
# Hargreaves simplified using mean temperature
# PET (mm/year) ≈ 0.0023 * (Tmean + 17.8) * 365
df["PET"] = 0.0023 * (df["temp_c"] + 17.8) * 365

# --- 2️⃣ Compute runoff (rain - PET) with runoff coefficient ---
runoff_coeff = 0.45  # typical Volta Basin range 0.3–0.6
df["runoff_mm"] = runoff_coeff * np.maximum(0, df["rain_mm"] - df["PET"])

# --- 3️⃣ Simple reservoir storage model ---
df["storage"] = df["runoff_mm"].cumsum() - df["runoff_mm"].mean() * np.arange(len(df))
df["storage"] = df["storage"] - df["storage"].min()  # normalize to start at 0

# --- 4️⃣ Estimate potential generation (scaled) ---
scale_factor = df["generation_smooth"].mean() / df["storage"].mean()
df["simulated_generation"] = df["storage"] * scale_factor

# --- 5️⃣ Evaluate performance ---
corr = df["simulated_generation"].corr(df["generation_smooth"])
bias = (df["simulated_generation"] - df["generation_smooth"]).mean()
rmse = np.sqrt(((df["simulated_generation"] - df["generation_smooth"])**2).mean())

print("\n📊 Model Performance:")
print(f"   • Correlation (r): {corr:.3f}")
print(f"   • Bias: {bias:.2f} GWh")
print(f"   • RMSE: {rmse:.2f} GWh")

# --- 6️⃣ Save results ---
df.to_csv("./outputs/hydro_model_results.csv", index=False)

# --- 7️⃣ Plot observed vs simulated ---
plt.figure(figsize=(10,5))
plt.plot(df["year"], df["generation_smooth"], label="Observed Generation", color="blue")
plt.plot(df["year"], df["simulated_generation"], label="Simulated Generation", color="green", linestyle="--")
plt.title("Observed vs Simulated Hydropower Generation (Conceptual Model)")
plt.xlabel("Year")
plt.ylabel("Generation (GWh)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./outputs/hydro_model_comparison.png", dpi=300)
plt.close()

# --- 8️⃣ Plot rainfall–runoff relationship ---
plt.figure(figsize=(7,5))
plt.scatter(df["rain_mm"], df["runoff_mm"], color="teal")
plt.title("Rainfall–Runoff Relationship (Volta Basin)")
plt.xlabel("Annual Rainfall (mm)")
plt.ylabel("Estimated Runoff (mm)")
plt.grid(True)
plt.tight_layout()
plt.savefig("./outputs/rain_runoff_relation.png", dpi=300)
plt.close()

print("\n✅ Hydrology model complete!")
print("Outputs saved in ./outputs/:")
print(" - hydro_model_results.csv")
print(" - hydro_model_comparison.png")
print(" - rain_runoff_relation.png")
