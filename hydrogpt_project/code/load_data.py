# File: code/load_data.py
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Load GEBCO
gebco_file = 'data/gebco_2025_n27.133_s10.29_w-88.05_e-59.033_geotiff.tif'  # Adjust path
with rasterio.open(gebco_file) as src:
    gebco_data = src.read(1).astype(float)
    profile = src.profile

print(f"GEBCO loaded: shape={gebco_data.shape}, dtype={gebco_data.dtype}")
print(f"Depth range: {np.nanmin(gebco_data):.0f}m to {np.nanmax(gebco_data):.0f}m")
print(f"Mean depth: {np.nanmean(gebco_data):.0f}m")

# Visualize
plt.figure(figsize=(10, 8))
plt.imshow(gebco_data, cmap='viridis', vmin=-5000, vmax=100)
plt.colorbar(label='Depth (m)')
plt.title('GEBCO 2024 Bathymetry')
plt.savefig('results/gebco_preview.png', dpi=100)
print("✓ Preview saved to results/gebco_preview.png")