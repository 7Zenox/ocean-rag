# code/llm_wrapper.py
import os
import json
import numpy as np
from llm_config import LLM_CONFIG

try:
    from google import genai
except ImportError:
    genai = None


def call_llm(prompt: str) -> str:
    """Call Google Gemini and return raw text."""
    if LLM_CONFIG["provider"] == "gemini" and genai is not None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=LLM_CONFIG["model"],
            contents=prompt,
        )
        return resp.text
    else:
        # Safe default for dev if Gemini not available
        return '{"threshold": 60.0, "window": 5}'


def parse_rule(
    json_str: str,
    default_threshold: float = 60.0,
    default_window: int = 5,
) -> tuple[float, int]:
    """Parse {"threshold": float, "window": int} from LLM output."""
    try:
        data = json.loads(json_str)
        thr = float(data.get("threshold", default_threshold))
        win = int(data.get("window", default_window))
    except Exception:
        thr = default_threshold
        win = default_window

    # Clamp to sane ranges
    thr = max(10.0, min(thr, 300.0))
    if win < 3:
        win = 3
    if win % 2 == 0:
        win += 1  # make odd

    return thr, win


def apply_llm_rule(corrupted: np.ndarray, threshold: float, window: int) -> np.ndarray:
    """Apply a spike-removal rule chosen by the LLM."""
    from scipy.ndimage import median_filter

    tile = corrupted.copy()
    med = median_filter(tile, size=window)
    diff = tile - med
    mask = np.abs(diff) > threshold

    corrected = tile.copy()
    corrected[mask] = med[mask]
    return corrected


def llm_correct_tile(corrupted: np.ndarray, tile_id: int) -> np.ndarray:
    """
    Real LLM-based corrector:
    - Send tile-level statistics to Gemini.
    - Get JSON with threshold+window.
    - Apply that rule to the tile.
    """
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

Goal:
Propose a conservative rule to remove obvious acoustic spikes while
preserving real bathymetric structure.

Respond with ONLY a JSON object, no extra text, with fields:
- "threshold": float, depth difference (m) used to flag spikes
- "window": odd int (3,5,7,...) for the median-filter window size

Example:
{{"threshold": 60.0, "window": 5}}
""".strip()

    raw = call_llm(prompt)
    thr, win = parse_rule(raw)
    corrected = apply_llm_rule(corrupted, threshold=thr, window=win)
    return corrected
