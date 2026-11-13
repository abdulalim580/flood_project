# Adaptive Flood Detection and Confidence Mapping
# Input  : Sentinel-2 Green (B03) and NIR (B08) bands
# Output : Flood map, confidence map, NDWI map, and area estimate

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from pathlib import Path
from skimage.filters import threshold_otsu


# -------------------- SETTINGS --------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

# Sentinel-2 bands
GREEN_FILE = None
NIR_FILE = None

# Sentinel-2 B03 and B08 have a nominal 10 m pixel size.
# This is an assumption until the image metadata is verified.
PIXEL_SIZE = 10.0

OUTPUT_DIR.mkdir(exist_ok=True)


# -------------------- FIND INPUT FILES --------------------

def find_band(band_name):
    """Find a downloaded Sentinel-2 band inside the data folder."""

    possible_files = list(DATA_DIR.rglob("*.jp2"))
    possible_files += list(DATA_DIR.rglob("*.JP2"))

    for file in possible_files:
        if band_name.lower() in file.name.lower():
            return file

    raise FileNotFoundError(
        f"Cannot find {band_name} JP2 file inside the data folder."
    )


GREEN_FILE = find_band("B03")
NIR_FILE = find_band("B08")

print("Green band:", GREEN_FILE)
print("NIR band:", NIR_FILE)


# -------------------- LOAD SATELLITE IMAGES --------------------

def load_band(file_path):
    """Read one satellite band and return its image and metadata."""

    with rasterio.open(file_path) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    return image.astype(np.float32), transform, crs, nodata


green, green_transform, green_crs, green_nodata = load_band(
    GREEN_FILE
)

nir, nir_transform, nir_crs, nir_nodata = load_band(
    NIR_FILE
)


# -------------------- CHECK INPUT DATA --------------------

if green.shape != nir.shape:
    raise ValueError(
        "Green and NIR images have different dimensions. "
        "Use matching Sentinel-2 bands."
    )

if green_crs != nir_crs or green_transform != nir_transform:
    raise ValueError(
        "The Green and NIR images do not have matching "
        "geospatial information. Align the images before continuing."
    )

if green.size == 0:
    raise ValueError("The satellite images are empty.")

# Ignore invalid pixels and pixels containing no data.
valid = np.isfinite(green) & np.isfinite(nir)

if green_nodata is not None:
    valid &= green != green_nodata

if nir_nodata is not None:
    valid &= nir != nir_nodata

valid &= (green != 0) | (nir != 0)

if not np.any(valid):
    raise ValueError("No valid satellite pixels were found.")


# -------------------- CALCULATE NDWI --------------------

# NDWI = (Green - NIR) / (Green + NIR)
# Positive NDWI values can indicate water.

denominator = green + nir

ndwi = np.zeros(green.shape, dtype=np.float32)

safe_pixels = valid & (np.abs(denominator) > 1e-6)

ndwi[safe_pixels] = (
    (green[safe_pixels] - nir[safe_pixels])
    / denominator[safe_pixels]
)


# -------------------- DETECT POTENTIAL WATER --------------------

valid_values = ndwi[valid]

# Use a sample to make threshold calculation faster.
sample = valid_values[::100]

if sample.size < 2:
    raise ValueError("Not enough valid pixels to detect water.")

# Otsu automatically selects a threshold from the image values.
threshold = threshold_otsu(sample)

# Positive NDWI values are generally associated with water.
threshold = max(float(threshold), 0.0)

# Classify pixels above the threshold as potential water.
flood_map = valid & (ndwi > threshold)

print("\n--- FLOOD DETECTION RESULTS ---")
print("Image dimensions:", green.shape)
print("Valid pixels:", int(np.sum(valid)))
print("NDWI threshold:", round(threshold, 4))
print("Potential water pixels:", int(np.sum(flood_map)))


# -------------------- CALCULATE WATER AREA --------------------

total_valid_pixels = int(np.sum(valid))
water_pixels = int(np.sum(flood_map))

if total_valid_pixels > 0:
    water_percentage = (
        water_pixels / total_valid_pixels
    ) * 100
else:
    water_percentage = 0.0

# Estimate area using the assumed 10 m pixel size.
pixel_area = PIXEL_SIZE * PIXEL_SIZE

water_area_km2 = (
    water_pixels * pixel_area
) / 1_000_000

print("Potential water percentage:",
      round(water_percentage, 2), "%")

print("Estimated water area:",
      round(water_area_km2, 4), "km²")

if green_crs is None:
    print(
        "WARNING: CRS is missing. "
        "The map location and area are not geospatially validated."
    )
else:
    print("Coordinate reference system:", green_crs)


# -------------------- CREATE CONFIDENCE MAP --------------------

# This is a relative confidence score based on distance
# from the NDWI threshold, not a calibrated probability.

# confidence = np.zeros(green.shape, dtype=np.float32)

# distance = np.abs(ndwi[valid] - threshold)

# scale = max(
#     float(np.percentile(distance, 95)),
#     1e-6
# )

# confidence[valid] = np.clip(
#     distance / scale,
#     0,
#     1
# )
# -------------------- CREATE SMALL PREVIEW IMAGES --------------------

# Use every 10th pixel for displaying images.
# This reduces memory usage without changing the original satellite data.

step = 10

small_ndwi = ndwi[::step, ::step]
small_valid = valid[::step, ::step]
small_flood = flood_map[::step, ::step]

# -------------------- CONFIDENCE MAP --------------------

# Calculate confidence only for the smaller preview image.
distance = np.abs(small_ndwi[small_valid] - threshold)

if distance.size > 0:
    scale = max(float(np.percentile(distance, 95)), 1e-6)
else:
    scale = 1e-6

confidence = np.zeros(small_ndwi.shape, dtype=np.float32)

confidence[small_valid] = np.clip(
    np.abs(small_ndwi[small_valid] - threshold) / scale,
    0,
    1
)

# -------------------- SAVE FLOOD MAP --------------------

plt.figure(figsize=(7, 6))
plt.imshow(small_flood, cmap="Blues")
plt.title("Potential Water / Flood Map")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "flood_map.png", dpi=150)
plt.close()

# -------------------- SAVE CONFIDENCE MAP --------------------

plt.figure(figsize=(7, 6))
plt.imshow(confidence, cmap="viridis", vmin=0, vmax=1)
plt.colorbar(label="Relative confidence")
plt.title("Water Classification Confidence Map")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confidence_map.png", dpi=150)
plt.close()

# -------------------- SAVE NDWI MAP --------------------

plt.figure(figsize=(7, 6))
plt.imshow(
    np.where(small_valid, small_ndwi, np.nan),
    cmap="RdYlBu",
    vmin=-1,
    vmax=1
)
plt.colorbar(label="NDWI")
plt.title("Normalized Difference Water Index")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ndwi_map.png", dpi=150)
plt.close()

# -------------------- FINISHED --------------------

print("\nOutput images saved in:", OUTPUT_DIR.resolve())
print("1. flood_map.png")
print("2. confidence_map.png")
print("3. ndwi_map.png")
print("\nFlood detection completed.")


# -------------------- SAVE FLOOD MAP --------------------

plt.figure(figsize=(7, 6))

plt.imshow(flood_map, cmap="Blues")

plt.title("Potential Water / Flood Map")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "flood_map.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# -------------------- SAVE CONFIDENCE MAP --------------------

plt.figure(figsize=(7, 6))

plt.imshow(
    confidence,
    cmap="viridis",
    vmin=0,
    vmax=1
)

plt.colorbar(label="Relative confidence")
plt.title("Water Classification Confidence Map")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confidence_map.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# -------------------- SAVE NDWI MAP --------------------

plt.figure(figsize=(7, 6))

plt.imshow(
    np.where(valid, ndwi, np.nan),
    cmap="RdYlBu",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="NDWI")
plt.title("Normalized Difference Water Index")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ndwi_map.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# -------------------- FINISHED --------------------

print("\nOutput images saved in:", OUTPUT_DIR.resolve())

print("1. flood_map.png")
print("2. confidence_map.png")
print("3. ndwi_map.png")

print("\nFlood detection completed.")