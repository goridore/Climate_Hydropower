import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

# ==============================
# 1️⃣ Load datasets
# ==============================

print("📥 Loading datasets...")

# Monthly climate data (rain, temp, spei)
climate = pd.read_csv("./outputs/volta_monthly_climate.csv", parse_dates=["date"])
# Hydropower generation data (filled)
hydro = pd.read_csv("./outputs/hydropower_filled.csv")

# ==============================
# 2️⃣ Aggregate climate data to annual scale
# ==============================

print("📊 Aggregating climate data to annual scale...")

climate["year"] = climate["date"].dt.year
annual_climate = climate.groupby("year").agg({
    "rain_mm": "sum",      # total annual rainfall (mm)
    "temp_c": "mean",      # average annual temperature (°C)
    "spei": "mean"         # mean annual SPEI
}).reset_index()

annual_climate.to_csv("./outputs/volta_annual_climate.csv", index=False)
print("✅ Saved annual climate data:", annual_climate.shape)

# ==============================
# 3️⃣ Merge with hydropower dataset
# ==============================

print("🔗 Merging climate and hydropower datasets...")

merged = pd.merge(hydro, annual_climate, on="year", how="inner")
merged.to_csv("./outputs/hydro_climate_merged.csv", index=False)
print("✅ Saved merged hydro-climate dataset:", merged.shape)
print(merged.head())

# ==============================
# 4️⃣ Correlation analysis
# ==============================

print("📈 Performing correlation analysis...")

corr = merged[["generation_smooth", "rain_mm", "temp_c", "spei"]].corr(method="pearson")

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=False)
plt.title("Correlation Between Hydropower and Climate Variables")
plt.tight_layout()
plt.savefig("./outputs/hydro_climate_correlation.png", dpi=300)
plt.close()

print("✅ Correlation heatmap saved to ./outputs/hydro_climate_correlation.png")
print(corr)

# ==============================
# 5️⃣ Multiple Linear Regression (OLS)
# ==============================

print("📊 Running Multiple Linear Regression...")

# Define features and target
X = merged[["rain_mm", "temp_c", "spei"]]
y = merged["generation_smooth"]

# Normalize predictors (optional but improves stability)
X = (X - X.mean()) / X.std()

# Add constant for OLS intercept
X = sm.add_constant(X)

# Fit regression model
model = sm.OLS(y, X).fit()

# Print summary to console
print(model.summary())

# Save regression summary to text file
with open("./outputs/hydro_regression_summary.txt", "w") as f:
    f.write(model.summary().as_text())
print("✅ Regression summary saved to ./outputs/hydro_regression_summary.txt")

# ==============================
# 6️⃣ Plot observed vs predicted
# ==============================

print("📉 Generating regression fit plot...")

merged["predicted_gen"] = model.predict(X)

plt.figure(figsize=(10,5))
plt.plot(merged["year"], merged["generation_smooth"], label="Observed", color="blue")
plt.plot(merged["year"], merged["predicted_gen"], label="Predicted", color="red", linestyle="--")
plt.title("Observed vs Predicted Hydropower Generation (1981–2023)")
plt.ylabel("Generation (GWh)")
plt.xlabel("Year")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./outputs/hydro_generation_regression_fit.png", dpi=300)
plt.close()

print("✅ Regression fit plot saved to ./outputs/hydro_generation_regression_fit.png")

# ==============================
# 7️⃣ Wrap-up
# ==============================

print("\n🎯 ANALYSIS COMPLETE!")
print("Generated files:")
print(" - volta_annual_climate.csv")
print(" - hydro_climate_merged.csv")
print(" - hydro_climate_correlation.png")
print(" - hydro_regression_summary.txt")
print(" - hydro_generation_regression_fit.png")

# ==============================
# 8️⃣ Trend Analysis (Mann–Kendall Test)
# ==============================
import pymannkendall as mk

print("\n🔍 Running Mann–Kendall trend tests...")

# --- Rainfall trend ---
rain_trend = mk.original_test(annual_climate["rain_mm"])
print("\n🌧 Rainfall Trend Test Results:")
print(rain_trend)

# --- Temperature trend ---
temp_trend = mk.original_test(annual_climate["temp_c"])
print("\n🌡 Temperature Trend Test Results:")
print(temp_trend)

# --- Save results ---
with open("./outputs/mann_kendall_results.txt", "w") as f:
    f.write("🌧 Rainfall Trend Test:\n")
    f.write(str(rain_trend) + "\n\n")
    f.write("🌡 Temperature Trend Test:\n")
    f.write(str(temp_trend) + "\n")

print("✅ Mann–Kendall trend results saved to ./outputs/mann_kendall_results.txt")

# --- Optional: visualize trend lines ---
plt.figure(figsize=(10,4))
plt.plot(annual_climate["year"], annual_climate["rain_mm"], color="blue", label="Rainfall")
plt.title("Annual Rainfall Trend (1981–2023)")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.grid(True)
z = np.polyfit(annual_climate["year"], annual_climate["rain_mm"], 1)
plt.plot(annual_climate["year"], np.polyval(z, annual_climate["year"]), color="red", linestyle="--", label="Trend")
plt.legend()
plt.tight_layout()
plt.savefig("./outputs/mk_trend_rainfall.png", dpi=300)
plt.close()

plt.figure(figsize=(10,4))
plt.plot(annual_climate["year"], annual_climate["temp_c"], color="orange", label="Temperature")
plt.title("Annual Temperature Trend (1981–2023)")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.grid(True)
z = np.polyfit(annual_climate["year"], annual_climate["temp_c"], 1)
plt.plot(annual_climate["year"], np.polyval(z, annual_climate["year"]), color="red", linestyle="--", label="Trend")
plt.legend()
plt.tight_layout()
plt.savefig("./outputs/mk_trend_temperature.png", dpi=300)
plt.close()

print("✅ Trend plots saved to ./outputs/")
