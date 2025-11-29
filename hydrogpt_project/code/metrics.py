# File: code/metrics.py
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

class BathymetryMetrics:
    """Compute evaluation metrics for bathymetric corrections"""
    
    def __init__(self, ground_truth, predicted):
        self.gt = ground_truth
        self.pred = predicted
    
    def rmse(self):
        """Root Mean Squared Error - main metric (in meters)"""
        return np.sqrt(np.mean((self.gt - self.pred) ** 2))
    
    def mae(self):
        """Mean Absolute Error (in meters)"""
        return np.mean(np.abs(self.gt - self.pred))
    
    def precision_recall_f1(self, corrupted, artifact_threshold=50):
        """
        Artifact detection metrics.
        Treats as binary classification:
        - Positive = has artifact (error > threshold)
        - Predict positive = LLM corrected it
        """
        original_error = np.abs(corrupted - self.gt)
        has_artifact = original_error > artifact_threshold
        
        corrected_error = np.abs(self.pred - self.gt)
        improved = original_error - corrected_error
        predicted_artifact_fixed = improved > 10
        
        # TP, FP, FN, TN
        tp = np.sum(has_artifact & predicted_artifact_fixed)
        fp = np.sum(~has_artifact & predicted_artifact_fixed)
        fn = np.sum(has_artifact & ~predicted_artifact_fixed)
        tn = np.sum(~has_artifact & ~predicted_artifact_fixed)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def processing_speed(self, time_seconds, tile_size=(512, 512)):
        """Pixels corrected per second"""
        total_pixels = tile_size * tile_size
        return total_pixels / time_seconds if time_seconds > 0 else 0

# Test it
if __name__ == "__main__":
    # Dummy test
    clean = np.random.rand(100, 100) * 100
    corrupted = clean + np.random.normal(0, 10, (100, 100))
    
    m = BathymetryMetrics(clean, corrupted)
    print(f"RMSE: {m.rmse():.2f}m")
    print(f"MAE: {m.mae():.2f}m")
    print(f"Speed: {m.processing_speed(2.0):.0f} pixels/sec")
