import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.geometry import mapping

# Load datasets
spei = xr.open_dataset('./data/spei12.nc')
basin = gpd.read_file('./data/volta_basin.shp')

# Ensure correct coordinate names
if 'lat' in spei.coords and 'lon' in spei.coords:
    spei = spei.rename({'lat': 'latitude', 'lon': 'longitude'})

# Assign CRS explicitly — rioxarray needs this
spei = spei.rio.write_crs("EPSG:4326")

# Optional: make sure latitude is descending (some datasets flip)
if spei.latitude[0] < spei.latitude[-1]:
    spei = spei.sortby('latitude', ascending=False)

spei_clip = spei.rio.clip(basin.geometry.apply(mapping), basin.crs)
spei_clip.load()  # force compute

print("SPEI mean:", spei_clip["spei"].mean().values)
