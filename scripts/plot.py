import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


rain = xr.open_dataset("./outputs/volta_rain.nc")
temp = xr.open_dataset("./outputs/volta_temp.nc")
spei = xr.open_dataset("./outputs/volta_spei.nc")

print("Rain:", rain["time"].min().values, "→", rain["time"].max().values)
print("Temp:", temp["time"].min().values, "→", temp["time"].max().values)
print("SPEI:", spei["time"].min().values, "→", spei["time"].max().values)


rain_ts = rain["precip"].mean(dim=["latitude", "longitude"])
temp_ts = temp["t2m"].mean(dim=["latitude", "longitude"]) - 273.15
spei_ts = spei["spei"].mean(dim=["latitude", "longitude"])

rain_df = rain_ts.to_dataframe(name="rain_mm").reset_index()
temp_df = temp_ts.to_dataframe(name="temp_c").reset_index()
spei_df = spei_ts.to_dataframe(name="spei").reset_index()

# Align overlapping period
start = max(rain_df["time"].min(), temp_df["time"].min(), spei_df["time"].min())
end = min(rain_df["time"].max(), temp_df["time"].max(), spei_df["time"].max())

rain_df = rain_df[(rain_df["time"] >= start) & (rain_df["time"] <= end)]
temp_df = temp_df[(temp_df["time"] >= start) & (temp_df["time"] <= end)]
spei_df = spei_df[(spei_df["time"] >= start) & (spei_df["time"] <= end)]


for df in [rain_df, temp_df, spei_df]:
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()


merged = pd.merge(rain_df, temp_df, on="time", how="inner")
merged = pd.merge(merged, spei_df, on="time", how="inner").rename(columns={"time": "date"})

merged.to_csv("./outputs/volta_monthly_climate.csv", index=False)
print("✅ Saved merged climate data:", merged.shape)


plt.figure(figsize=(10,4))
plt.plot(merged["date"], merged["rain_mm"], color="blue")
plt.title("Volta Basin Monthly Rainfall (1981–2023)")
plt.ylabel("Rainfall (mm)")
plt.grid(True)
plt.tight_layout()
plt.savefig("./outputs/volta_rain_trend.png", dpi=300)
plt.close()

plt.figure(figsize=(10,4))
plt.plot(merged["date"], merged["temp_c"], color="red")
plt.title("Volta Basin Monthly Temperature (1981–2023)")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.tight_layout()
plt.savefig("./outputs/volta_temp_trend.png", dpi=300)
plt.close()

plt.figure(figsize=(10,4))
plt.plot(merged["date"], merged["spei"], color="green")
plt.axhline(0, color="black", lw=1)
plt.title("Volta Basin 12-Month SPEI (1981–2023)")
plt.ylabel("SPEI")
plt.grid(True)
plt.tight_layout()
plt.savefig("./outputs/volta_spei_trend.png", dpi=300)
plt.close()

print("✅ Saved all three plots to ./outputs/")
