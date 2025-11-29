# code/llm_systems.py
import numpy as np
from baselines import BaselineCorrections

class LLMSystems:
    """
    LLM variants. This week they are heuristic stubs.
    Later you replace internals with real LLM calls.
    """

    @staticmethod
    def system_baseline(corrupted: np.ndarray, context=None) -> np.ndarray:
        # Same as Baseline 1 – used as lower bound
        return BaselineCorrections.heuristic_spike_removal(corrupted)

    @staticmethod
    def system_lite_llm(corrupted: np.ndarray, context=None) -> np.ndarray:
        """
        LLM-lite stub:
        - Detect 'spiky' pixels (large deviation from local median).
        - Replace them with local median.
        - Slightly smarter thresholds than baseline.
        """
        from scipy.ndimage import median_filter

        tile = corrupted.copy()
        med = median_filter(tile, size=5)
        diff = tile - med

        # dynamic threshold based on local variability
        thresh = np.std(diff) * 1.5
        mask = np.abs(diff) > thresh
        tile[mask] = med[mask]
        return tile

    @staticmethod
    def system_full_llm(corrupted: np.ndarray, context=None) -> np.ndarray:
        """
        Full LLM stub:
        - Same as LLM-lite, plus a 'physics' constraint:
          keep large-scale structure by blending with original depths.
        """
        lite = LLMSystems.system_lite_llm(corrupted, context)

        # Blend: preserve low-frequency structure via smoothing original
        from scipy.ndimage import gaussian_filter
        low_freq = gaussian_filter(corrupted, sigma=10)

        # Weighted combination (could be tuned)
        alpha = 0.7  # weight on LLM-lite result
        full = alpha * lite + (1 - alpha) * low_freq
        return full
