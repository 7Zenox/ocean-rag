# File: code/corrupt_tiles.py
import numpy as np
import os

def corrupt_tile(clean_tile, tile_id, corruption_level='medium'):
    """
    Introduce realistic bathymetric artifacts
    """
    corrupted = clean_tile.copy().astype(float)
    h, w = corrupted.shape
    
    if corruption_level == 'medium':
        # Artifact 1: Random spikes (5% of pixels, 50-200m error)
        spike_rate = 0.05
        spike_mask = np.random.random((h, w)) < spike_rate
        spike_mags = np.random.uniform(50, 200, spike_mask.sum())
        corrupted[spike_mask] += spike_mags
        
        # Artifact 2: Bias patches (stripes, ±30m)
        for stripe_start in range(0, h, 50):
            bias = np.random.uniform(-30, 30)
            corrupted[stripe_start:min(stripe_start+50, h), :] += bias
        
        # Artifact 3: Gaussian noise (~5% of depth)
        noise_std = np.abs(clean_tile) * 0.05
        noise = np.random.normal(0, noise_std)
        corrupted += noise
        
        return corrupted, spike_mask
    
    return corrupted, spike_mask

# Corrupt all tiles
for tile_id in range(1, 21):
    clean = np.load(f'benchmark_data/clean_tile_{tile_id:02d}.npy')
    corrupted, mask = corrupt_tile(clean, tile_id)
    
    np.save(f'benchmark_data/corrupted_tile_{tile_id:02d}.npy', corrupted)
    np.save(f'benchmark_data/groundtruth_mask_{tile_id:02d}.npy', mask)
    
    print(f"Created corrupted tile {tile_id}/20 - "
          f"RMSE (corrupted vs clean): {np.sqrt(np.mean((corrupted - clean)**2)):.1f}m")

print("✓ All 20 tiles corrupted and saved!")
