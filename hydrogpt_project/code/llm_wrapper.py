# code/llm_wrapper.py
import os
import numpy as np
from llm_config import LLM_CONFIG

try:
    from google import genai
except ImportError:
    genai = None

def call_llm(prompt: str) -> str:
    """Wrapper for Google Gemini API."""
    if LLM_CONFIG["provider"] == "gemini" and genai is not None:
        api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyBfdfQQ_UWVlzyVMy00yTbCVabiEBVF3FQ")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=LLM_CONFIG["model"],
            contents=prompt
        )
        return response.text
    else:
        # For now, just echo prompt (debugging stub)
        return "STUB_RESPONSE"

def llm_correct_tile(corrupted: np.ndarray, tile_id: int) -> np.ndarray:
    """
    Interface that a real LLM-based corrector will satisfy.
    For Week 2, you don't need to use this yet.
    """
    # Build a compact textual representation (stats only, not full grid)
    depth_min = float(np.nanmin(corrupted))
    depth_max = float(np.nanmax(corrupted))
    depth_mean = float(np.nanmean(corrupted))

    prompt = f"""
You are assisting in cleaning noisy bathymetry (ocean depth) data.

Tile ID: {tile_id}
Depth statistics (meters):
- min: {depth_min:.1f}
- max: {depth_max:.1f}
- mean: {depth_mean:.1f}

Suggest a simple rule to remove spikes and preserve realistic depth ranges.
Return JSON with fields:
- 'threshold'
- 'window'
"""
    _ = call_llm(prompt)
    # For now, ignore output and fallback to stub logic.
    from llm_systems import LLMSystems
    return LLMSystems.system_lite_llm(corrupted)
