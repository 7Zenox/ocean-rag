# File: code/create_tiles.py
import numpy as np
import rasterio

def split_into_tiles(gebco_data, tile_size=512, num_tiles=20):
    """Split large GEBCO grid into manageable tiles"""
    h, w = gebco_data.shape
    tiles = []
    
    # Sample tiles to cover different depth ranges
    np.random.seed(42)
    for i in range(num_tiles):
        # Random position (avoid edges)
        y = np.random.randint(0, h - tile_size)
        x = np.random.randint(0, w - tile_size)
        
        tile = gebco_data[y:y+tile_size, x:x+tile_size]
        tiles.append(tile)
    
    return tiles

# Load and tile
gebco_file = 'data/gebco_2025_n27.133_s10.29_w-88.05_e-59.033_geotiff.tif'
with rasterio.open(gebco_file) as src:
    gebco_data = src.read(1).astype(float)

tiles = split_into_tiles(gebco_data, tile_size=512, num_tiles=20)

# Save tiles
import os
os.makedirs('benchmark_data', exist_ok=True)

for i, tile in enumerate(tiles):
    np.save(f'benchmark_data/clean_tile_{i+1:02d}.npy', tile)
    print(f"Saved clean_tile_{i+1:02d}.npy (shape={tile.shape}, range={tile.min():.0f}-{tile.max():.0f}m)")

print(f"✓ Created {len(tiles)} tiles in benchmark_data/")
