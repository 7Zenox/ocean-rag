# File: code/baselines.py
import numpy as np
from scipy.ndimage import median_filter
from scipy.interpolate import griddata

class BaselineCorrections:
    """Reference methods to compare against"""
    
    @staticmethod
    def heuristic_spike_removal(corrupted_depth, threshold=50, window=5):
        """
        Baseline 1: Median filter + outlier removal
        """
        # First pass: median filter
        filtered = median_filter(corrupted_depth, size=window)
        
        # Second pass: replace outliers
        corrected = corrupted_depth.copy()
        diff = np.abs(filtered - corrupted_depth)
        outliers = diff > threshold
        corrected[outliers] = filtered[outliers]
        
        return corrected
    
    @staticmethod
    def interpolation_based(corrupted_depth, mask=None):
        """
        Baseline 2: Interpolate corrupted regions
        """
        if mask is None:
            # Flag pixels as outliers if >2 std deviations from median
            median = np.median(corrupted_depth)
            std = np.std(corrupted_depth)
            mask = np.abs(corrupted_depth - median) > 2 * std
        
        h, w = corrupted_depth.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Interpolate
        valid = ~mask
        corrected = griddata(
            (y[valid], x[valid]),
            corrupted_depth[valid],
            (y, x),
            method='linear'
        )
        
        # Fill any remaining NaN with original
        nan_mask = np.isnan(corrected)
        corrected[nan_mask] = corrupted_depth[nan_mask]
        
        return corrected

# Test on first tile
if __name__ == "__main__":
    from metrics import BathymetryMetrics
    
    clean = np.load('benchmark_data/clean_tile_01.npy')
    corrupted = np.load('benchmark_data/corrupted_tile_01.npy')
    
    b1 = BaselineCorrections.heuristic_spike_removal(corrupted)
    b2 = BaselineCorrections.interpolation_based(corrupted)
    
    print("Baseline Results (Tile 01):")
    print(f"  Baseline 1 (Heuristic) RMSE: {BathymetryMetrics(clean, b1).rmse():.2f}m")
    print(f"  Baseline 2 (Interpolation) RMSE: {BathymetryMetrics(clean, b2).rmse():.2f}m")
    print(f"  Corrupted (no correction) RMSE: {BathymetryMetrics(clean, corrupted).rmse():.2f}m")
