import geopandas as gpd

# Load the HydroBASINS shapefile
basins = gpd.read_file("./data/hybas_af_lev06_v1c.shp")

# Define a bounding box around the Volta Basin region (lon_min, lat_min, lon_max, lat_max)
# Covers most of Ghana, Burkina Faso, Togo, Benin, and Ivory Coast east
volta_bbox = {
    "minx": -3.5,   # west of Ghana
    "maxx": 2.5,    # east of Togo/Benin
    "miny": 5.0,    # southern Ghana
    "maxy": 14.5    # northern Burkina Faso
}

# Filter basins that intersect this bounding box
volta = basins.cx[volta_bbox["minx"]:volta_bbox["maxx"], volta_bbox["miny"]:volta_bbox["maxy"]]

# Save the filtered subset
volta.to_file("./outputs/volta_basin.shp")

print(f"Saved {len(volta)} polygons likely within the Volta Basin region.")

# # Step 1: Load the big HydroBASINS shapefile
# basins = gpd.read_file("./data/hybas_af_lev06_v1c.shp")

# # Step 2: Filter only Volta basin polygons
# volta = basins[basins['NAME'].str.contains("Volta", case=False, na=False)]

# # Step 3: Save just the Volta polygons
# volta.to_file("../data/volta_basin.shp")

# print(f"Saved {len(volta)} Volta polygons to ../data/volta_basin.shp")
