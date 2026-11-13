# make_sample_data.py
# Creates a fake "satellite scene" so you can test the project
# before downloading real Sentinel-2 images.

import numpy as np
import cv2

np.random.seed(7)
H, W = 500, 500

# true flood area (this becomes our ground truth)
truth = np.zeros((H, W), dtype=np.uint8)
cv2.ellipse(truth, (170, 260), (120, 70), 20, 0, 360, 1, -1)
cv2.ellipse(truth, (380, 140), (60, 40), -30, 0, 360, 1, -1)
cv2.line(truth, (0, 420), (500, 380), 1, 25)

# land: NIR is higher than green
green = np.random.normal(900, 80, (H, W))
nir = np.random.normal(2800, 250, (H, W))

# water: green is higher than NIR
water = truth == 1
green[water] = np.random.normal(1100, 100, water.sum())
nir[water] = np.random.normal(450, 120, water.sum())

# dark shadow patches that look a bit like water
for (x, y) in [(400, 380), (80, 60), (300, 450)]:
    cv2.circle(green, (x, y), 12, 950, -1)
    cv2.circle(nir, (x, y), 12, 1100, -1)

# a strip of "no data" at the edge
green[:, :6] = 0
nir[:, :6] = 0

green = np.clip(green, 0, 65535).astype(np.uint16)
nir = np.clip(nir, 0, 65535).astype(np.uint16)

cv2.imwrite("data/B03.tif", green)
cv2.imwrite("data/B08.tif", nir)
cv2.imwrite("data/ground_truth.png", truth * 255)
print("Sample data saved in the data/ folder")