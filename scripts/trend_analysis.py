# Trend_analysis_ghana.py
# Objective 1: Download CHIRPS, aggregate to Volta / dam catchments, run Mann-Kendall + Sen slope, plot.

# 0) Install (run once)
# pip install xarray rioxarray netCDF4 geopandas rasterio pymannkendall scipy matplotlib cartopy pyproj requests

import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # for rasterio backend
import geopandas as gpd
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy.stats import theilslopes
import warnings
warnings.filterwarnings("ignore")

# ========== USER CONFIG ==========
outdir = Path("trend_outputs")
outdir.mkdir(parents=True, exist_ok=True)

# bounding box for Ghana (loose) - adjust if you will mask by catchment shapefile
# longitudes: -3.3 .. 1.2   latitudes: 4.7 .. 11.2
ghana_bbox = dict(lon_min=-3.3, lon_max=1.2, lat_min=4.7, lat_max=11.2)

# CHIRPS netCDF location (global monthly). This is a direct file URL hosted by CHC (works with wget/curl).
# If the URL changes, see https://www.chc.ucsb.edu/data for the latest path.
chirps_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc"

# Where to save downloaded files
raw_dir = outdir / "raw"
raw_dir.mkdir(exist_ok=True)

chirps_local = raw_dir / "chirps-v2.0.monthly.nc"

# ========== 1) Download CHIRPS (if not present) ==========
if not chirps_local.exists():
    print("Downloading CHIRPS monthly netCDF (this may take a minute)...")
    # Better to use wget/curl; fallback to requests streaming if wget not available
    try:
        subprocess.check_call(["wget", "-O", str(chirps_local), chirps_url])
    except Exception:
        import requests
        with requests.get(chirps_url, stream=True) as r:
            r.raise_for_status()
            with open(chirps_local, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    print("Downloaded CHIRPS ->", chirps_local)
else:
    print("Found existing CHIRPS file:", chirps_local)

# ========== 2) Load CHIRPS and subset to Ghana bbox ==========
ds = xr.open_dataset(chirps_local)  # variable often 'precip' or 'precipitation' depending on file
# find variable name
var_name = [v for v in ds.data_vars][0]
print("CHIRPS var:", var_name)
precip = ds[var_name]  # expected units: mm/month
# select bbox and time (optional)
precip_gha = precip.sel(longitude=slice(ghana_bbox['lon_min'], ghana_bbox['lon_max']),
                        latitude=slice(ghana_bbox['lat_max'], ghana_bbox['lat_min']))  # note lat order
print(precip_gha)

# ========== 3) (Option A) Aggregate spatially over the whole Volta basin using a shapefile ==========
# Download / supply a basin shapefile. Recommended sources:
# - HydroBASINS for Africa: https://www.hydrosheds.org/products/hydrobasins
# - Volta-specific layers via IWLearn / VBA or HydroSHEDS.
# For convenience, you can supply a shapefile 'volta_basin.shp' in the 'shapefiles' folder.
shapefiles_dir = Path("shapefiles")
shapefiles_dir.mkdir(exist_ok=True)
volta_shp = shapefiles_dir / "volta_basin.shp"

if not volta_shp.exists():
    print("No local Volta shapefile found at", volta_shp)
    print("Please download a Volta/Volta-subbasins shapefile (HydroBASINS or IWLearn) and place it at:")
    print(volta_shp)
    # fallback: approximate Ghana bbox as mask if no shapefile
    use_bbox_mask = True
else:
    use_bbox_mask = False

def area_weighted_timeseries(da, mask_gdf=None):
    """
    da: xarray DataArray with lat/lon dims named 'latitude' and 'longitude' (or 'lat'/'lon').
    mask_gdf: geopandas GeoDataFrame in same CRS (EPSG:4326) to mask/clip.
    returns: pandas Series indexed by time with area-weighted mean (mm/month)
    """
    # standardize dimension names
    if 'lat' in da.dims and 'lon' in da.dims:
        da = da.rename({'lat':'latitude','lon':'longitude'})
    # reproject mask to lat/lon (should already be)
    if mask_gdf is None:
        # simple bbox average
        ts = da.mean(dim=['latitude','longitude']).to_series()
        return ts
    # compute mask raster aligned to da grid
    import rasterio
    from rasterio.features import geometry_mask
    lon = da['longitude'].values
    lat = da['latitude'].values
    res_lon = abs(lon[1]-lon[0])
    res_lat = abs(lat[1]-lat[0])
    transform = rasterio.transform.from_origin(lon.min() - res_lon/2, lat.max() + res_lat/2, res_lon, res_lat)
    shapes = [(geom, 1) for geom in mask_gdf.to_crs(epsg=4326).geometry]
    mask = geometry_mask([g for g in mask_gdf.geometry],
                         out_shape=(len(lat), len(lon)),
                         transform=transform,
                         invert=True)
    # mask has shape (rows=lat, cols=lon) - align indexes
    mask_xr = xr.DataArray(mask.astype(float), coords=[da.latitude, da.longitude], dims=['latitude','longitude'])
    da_masked = da.where(mask_xr==1)
    # area-weighted mean: cos(lat) weighting
    weights = np.cos(np.deg2rad(da_masked.latitude))
    weights = weights / weights.mean()
    ts = (da_masked * weights).mean(dim=['latitude','longitude']).to_series()
    return ts

if not use_bbox_mask:
    # load shapefile and select Volta polygon (if multiple)
    gdf = gpd.read_file(volta_shp)
    # if layer has many polygons, you may need to select by NAME or ID
    # Example: select polygon whose NAME contains 'Volta' (adjust to shapefile attributes)
    if len(gdf) > 1:
        # try to find a field name with 'name' or 'NAME'
        namecols = [c for c in gdf.columns if 'name' in c.lower()]
        if namecols:
            sel = gdf[gdf[namecols[0]].str.contains('Volta', case=False, na=False)]
            if len(sel) > 0:
                gdf = sel
    ts_precip = area_weighted_timeseries(precip_gha, mask_gdf=gdf)
else:
    ts_precip = area_weighted_timeseries(precip_gha, mask_gdf=None)

# keep monthly series, ensure datetime index
ts_precip.index = pd.to_datetime(ts_precip.index)
ts_precip = ts_precip.sort_index()
ts_precip.name = "precip_mm"

# Save raw timeseries
ts_precip.to_csv(outdir / "precip_timeseries_volta.csv")
print("Saved aggregated precipitation timeseries:", outdir / "precip_timeseries_volta.csv")

# ========== 4) Trend analysis: Mann-Kendall + Sen's slope ==========
# We'll use pymannkendall for MK test (handles autocorrelation options) and scipy.theilslopes for Sen's slope.

# 4a: Annual / seasonal aggregation if needed
precip_annual = ts_precip.resample("A").sum()  # total annual mm
precip_monthly = ts_precip  # monthly mm

# 4b: Mann-Kendall (original) on annual totals
mk_res_annual = mk.original_test(precip_annual.values)
print("Mann-Kendall (annual) result:", mk_res_annual)

# 4c: Seasonal Mann-Kendall on monthly series
# pymannkendall has seasonal_test which expects array shaped (n_years, 12) or a flat series with monthly frequency
try:
    mk_seasonal = mk.seasonal_test(precip_monthly.values, period=12)
    print("Seasonal MK:", mk_seasonal)
except Exception as e:
    print("Seasonal MK failed:", e)

# 4d: Sen's slope (Theil-Sen)
# theilslopes gives slope per unit index (we convert to mm/year where needed)
# For annual:
x = np.arange(len(precip_annual))
slope, intercept, lo_slope, hi_slope = theilslopes(precip_annual.values.astype(float), x, 0.95)
# slope is in mm per index-step; index-step=1 year here
print("Sen's slope (annual) = {:.3f} mm/year (95% CI: [{:.3f}, {:.3f}])".format(slope, lo_slope, hi_slope))

# For monthly series, get slope per month -> convert to mm/decade for readability
x_m = np.arange(len(precip_monthly))
slope_m, intercept_m, lo_m, hi_m = theilslopes(precip_monthly.values.astype(float), x_m, 0.95)
slope_mm_per_year = slope_m * 12
slope_mm_per_decade = slope_mm_per_year * 10
print("Sen's slope (monthly) ~ {:.3f} mm/decade".format(slope_mm_per_decade))

# ========== 5) Plots: timeseries, anomaly, MK visualization ==========
plt.figure(figsize=(10,4))
plt.plot(precip_monthly.index, precip_monthly.values, label="Monthly precip (mm)")
plt.plot(precip_annual.index, precip_annual.values/12, label="Annual mean (mm/month equivalent)", linewidth=2)
plt.title("Volta-aggregated precipitation")
plt.legend()
plt.savefig(outdir / "precip_timeseries.png", dpi=200)
plt.close()

# anomaly
clim = precip_monthly.groupby(precip_monthly.index.month).mean()
monthly_anom = precip_monthly - precip_monthly.index.month.map(clim.to_dict())
plt.figure(figsize=(10,4))
plt.plot(monthly_anom.index, monthly_anom.values)
plt.axhline(0, color='k', linewidth=0.6)
plt.title("Monthly precipitation anomalies (Volta aggregate)")
plt.savefig(outdir / "precip_anomaly.png", dpi=200)
plt.close()

# MK summary file
mk_summary = {
    "annual_mk_trend": mk_res_annual.trend,
    "annual_mk_p": mk_res_annual.p,
    "annual_sen_slope_mm_per_year": slope,
    "monthly_sen_slope_mm_per_decade": slope_mm_per_decade
}
pd.Series(mk_summary).to_csv(outdir / "mk_summary_volta.csv")
print("Saved MK summary ->", outdir / "mk_summary_volta.csv")

# ========== 6) OPTIONAL: grid-cell Mann-Kendall map (slow) ==========
# Run MK per grid-cell in the Ghana subset to create a spatial trend map (Sen's slope or MK p-value).
# NOTE: This can be computationally heavy. If you have >1e4 cells, consider parallelizing or sampling.

def grid_mann_kendall(da_slice):
    """da_slice: 1D array of length time"""
    try:
        res = mk.original_test(da_slice)
        trend = 0
        if res.trend == 'increasing':
            trend = 1
        elif res.trend == 'decreasing':
            trend = -1
        return (trend, res.p, res.Tau)
    except Exception:
        return (np.nan, np.nan, np.nan)

# Example: compute MK p-values for each cell (naive loop)
lons = precip_gha.longitude.values
lats = precip_gha.latitude.values
time = precip_gha.time.values
nlon = len(lons); nlat = len(lats)
p_map = np.full((nlat, nlon), np.nan)
tau_map = np.full((nlat, nlon), np.nan)

print("Computing cell-wise MK (this may take time)...")
for i in range(nlat):
    for j in range(nlon):
        series = precip_gha[:, i, j].values if precip_gha.ndim==3 else precip_gha[:, i, j].values
        if np.isnan(series).all():
            continue
        try:
            r = mk.original_test(series)
            p_map[i,j] = r.p
            tau_map[i,j] = r.Tau
        except Exception:
            p_map[i,j] = np.nan
            tau_map[i,j] = np.nan

# Save small netcdf or numpy maps
np.save(outdir / "mk_p_map.npy", p_map)
np.save(outdir / "mk_tau_map.npy", tau_map)
print("Saved MK raster arrays (npy) to", outdir)

# (You can convert maps to GeoTIFF using rasterio if you want a spatial plot)
# ========== END ==========
print("All done. Check directory:", outdir)
