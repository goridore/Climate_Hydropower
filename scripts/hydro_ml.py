"""
hydro_ml.py
------------
Objective 4: Machine Learning–based Prediction of Hydropower Generation

Input:
  - ./outputs/hydro_climate_merged.csv

Output:
  - ./outputs/ml_model_performance.csv
  - ./outputs/ml_predicted_vs_observed.png
  - ./outputs/ml_feature_importance.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import seaborn as sns

print("📥 Loading dataset...")
df = pd.read_csv("./outputs/hydro_climate_merged.csv")

# --- 1️⃣ Prepare features and target ---
X = df[["rain_mm", "temp_c", "spei"]]
y = df["generation_smooth"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 2️⃣ Define models ---
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "SVR": SVR(kernel="rbf", C=10, gamma="scale")
}

results = []

# --- 3️⃣ Train and evaluate each model ---
for name, model in models.items():
    print(f"\n🤖 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results.append({"Model": name, "R²": r2, "RMSE": rmse})
    print(f"✅ {name}: R²={r2:.3f}, RMSE={rmse:.2f}")

# --- 4️⃣ Save model performance ---
perf_df = pd.DataFrame(results)
perf_df.to_csv("./outputs/ml_model_performance.csv", index=False)
print("\n📊 Model performance saved to ./outputs/ml_model_performance.csv")
print(perf_df)

# --- 5️⃣ Best model (by R²) ---
best_model_name = perf_df.sort_values("R²", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]
print(f"\n🏆 Best model: {best_model_name}")

# --- 6️⃣ Predict full dataset with best model ---
best_preds = best_model.predict(X_scaled)
df["predicted_generation"] = best_preds

plt.figure(figsize=(10,5))
plt.plot(df["year"], df["generation_smooth"], label="Observed", color="blue")
plt.plot(df["year"], df["predicted_generation"], label="Predicted", color="orange", linestyle="--")
plt.title(f"Observed vs Predicted Hydropower Generation ({best_model_name})")
plt.xlabel("Year")
plt.ylabel("Generation (GWh)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./outputs/ml_predicted_vs_observed.png", dpi=300)
plt.close()

print("✅ Saved predicted vs observed plot to ./outputs/ml_predicted_vs_observed.png")

# --- 7️⃣ Feature importance (if model supports it) ---
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    features = ["rain_mm", "temp_c", "spei"]
    plt.figure(figsize=(6,4))
    sns.barplot(x=importances, y=features, palette="viridis")
    plt.title(f"{best_model_name} Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("./outputs/ml_feature_importance.png", dpi=300)
    plt.close()
    print("✅ Saved feature importance plot to ./outputs/ml_feature_importance.png")

print("\n🎯 MACHINE LEARNING ANALYSIS COMPLETE!")
print("Generated outputs:")
print(" - ml_model_performance.csv")
print(" - ml_predicted_vs_observed.png")
print(" - ml_feature_importance.png (if applicable)")
