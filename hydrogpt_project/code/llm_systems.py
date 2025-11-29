# code/llm_systems.py
import numpy as np
from baselines import BaselineCorrections
from scipy.ndimage import median_filter, gaussian_filter
from llm_wrapper import llm_correct_tile



class LLMSystems:
    """
    LLM variants.

    - system_baseline: pure heuristic baseline (no LLM).
    - system_lite_llm: heuristic LLM-inspired refinement (no API).
    - system_full_llm: heuristic physics-aware refinement (no API).
    - system_real_llm: uses Gemini to choose rule parameters.
    - system_real_full: future: combine real LLM rule + physics blend.
    """

    @staticmethod
    def system_baseline(corrupted: np.ndarray, context=None) -> np.ndarray:
        return BaselineCorrections.heuristic_spike_removal(corrupted)

    @staticmethod
    def system_lite_llm(corrupted: np.ndarray, context=None) -> np.ndarray:
        # Heuristic LLM-lite (what you already had)
        base = BaselineCorrections.heuristic_spike_removal(corrupted)
        med = median_filter(base, size=5)
        diff = base - med
        thresh = np.std(diff) * 2.5
        mask = np.abs(diff) > thresh
        lite = base.copy()
        lite[mask] = med[mask]
        return lite

    @staticmethod
    def system_full_llm(corrupted: np.ndarray, context=None) -> np.ndarray:
        # Heuristic physics-aware refinement (what you already had)
        lite = LLMSystems.system_lite_llm(corrupted, context)
        shallow_mask = corrupted > -1500
        smoothed = gaussian_filter(lite, sigma=1.5)

        full = lite.copy()
        full[shallow_mask] = smoothed[shallow_mask]

        deep_mask = ~shallow_mask
        if np.any(deep_mask):
            low_freq = gaussian_filter(corrupted, sigma=12)
            alpha = 0.8
            full[deep_mask] = alpha * lite[deep_mask] + (1.0 - alpha) * low_freq[deep_mask]

        return full

    @staticmethod
    def system_real_llm(corrupted: np.ndarray, context=None) -> np.ndarray:
        """
        Real LLM-guided correction:
        - Calls Gemini via llm_correct_tile to choose rule parameters.
        """
        tile_id = context.get("tile_id", 0) if context else 0
        return llm_correct_tile(corrupted, tile_id=tile_id)

    @staticmethod
    def system_real_full(corrupted: np.ndarray, context=None) -> np.ndarray:
        """
        Future extension:
        - Combine real LLM rule with physics-aware blending.
        For now, just call system_real_llm (same as above).
        """
        return LLMSystems.system_real_llm(corrupted, context)
