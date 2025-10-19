import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray
from shapely.geometry import mapping
from dask.diagnostics import ProgressBar

# Enable Dask lazy loading (avoid loading entire file into RAM)
xr.set_options(keep_attrs=True)

# Load Volta Basin shapefile
basin = gpd.read_file('./outputs/volta_basin.shp')

# Load datasets lazily
rain = xr.open_dataset('./data/chirps-v2.0.monthly.nc', chunks={"time": 12})
temp = xr.open_dataset('./data/era5_temp_ghana_1981_2024.nc', chunks={"time": 12})
temp = temp.rename({'valid_time': 'time'})
spei = xr.open_dataset('./data/spei12.nc', chunks={"time": 12})

# Assign CRS
rain = rain.rio.write_crs("EPSG:4326", inplace=True)
temp = temp.rio.write_crs("EPSG:4326", inplace=True)
if 'lat' in spei.coords and 'lon' in spei.coords:
    spei = spei.rename({'lat': 'latitude', 'lon': 'longitude'})
spei = spei.rio.write_crs("EPSG:4326")


if spei.latitude[0] < spei.latitude[-1]:
    spei = spei.sortby('latitude', ascending=False)
# Clip using Dask (lazy)
rain_clip = rain.rio.clip(basin.geometry.apply(mapping), basin.crs)
temp_clip = temp.rio.clip(basin.geometry.apply(mapping), basin.crs)
spei_clip = spei.rio.clip(basin.geometry.apply(mapping), basin.crs)
spei_clip.load()

# print("SPEI mean:", spei_clip["spei"].mean().values)
# print("SPEI time range:", spei_clip.time.min().values, "→", spei_clip.time.max().values)
# print("✅ SPEI successfully clipped to Volta Basin")
# print(spei_clip)

# data = spei_clip["spei"].values
# total = np.size(data)
# valid = np.count_nonzero(~np.isnan(data))
# print(f"Valid values: {valid}/{total} ({valid/total*100:.2f}% non-NaN)")
# Save clipped datasets efficiently
with ProgressBar():
    rain_clip.to_netcdf('./outputs/volta_rain.nc', compute=True)
    temp_clip.to_netcdf('./outputs/volta_temp.nc', compute=True)
    spei_clip.to_netcdf('./outputs/volta_spei.nc')


print("✅ Successfully clipped datasets to Volta Basin region.")
