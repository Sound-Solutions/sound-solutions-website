"""Inspect the actual delivered PNG pixels with Blender's bundled NumPy.

  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python-exit-code 1 -P brand/explore/astra/check_pixels.py

This is an artifact check, not a test of the renderer. It checks sizes, all four
transparent corners, nonblank content, centering, and approximately 8% margins.
It also reports visible-pixel contrast against both requested email backgrounds.
"""

import json
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ("1-soft-s", "2-cone", "3-fader", "4-pulse", "5-ripple")
results = []
for stem in NAMES:
    for suffix, size in (("", 1200), ("-sig", 252)):
        path = ROOT / f"{stem}{suffix}.png"
        assert path.is_file(), f"Missing {path.name}"
        img = bpy.data.images.load(str(path), check_existing=False)
        assert tuple(img.size) == (size, size), f"Wrong size: {path.name}"
        # Non-Color reads encoded PNG channel values, not linear-light values.
        img.colorspace_settings.name = "Non-Color"
        pixels = np.array(img.pixels[:], dtype=np.float32).reshape(size, size, 4)
        alpha = pixels[:, :, 3]
        for corner in (alpha[:4, :4], alpha[:4, -4:], alpha[-4:, :4], alpha[-4:, -4:]):
            assert np.all(corner == 0), f"Opaque corner in {path.name}"
        mask = alpha > 0.5
        count = int(mask.sum())
        assert count > size * size * 0.04, f"Blank or tiny: {path.name}"
        yy, xx = np.nonzero(mask)
        bounds = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
        margins = [bounds[0], bounds[1], size - 1 - bounds[2], size - 1 - bounds[3]]
        assert min(margins) >= size * 0.068, f"Insufficient margin: {path.name}"
        assert min(margins) <= size * 0.095, f"Mark too small: {path.name}"
        assert abs(margins[0] - margins[2]) <= 4, f"Not horizontally centered: {path.name}"
        assert abs(margins[1] - margins[3]) <= 4, f"Not vertically centered: {path.name}"
        rgb = pixels[:, :, :3][mask]
        spread = rgb.max(axis=1) - rgb.min(axis=1)
        neutral = rgb[spread < 0.025]
        assert len(neutral) > 0, f"No neutral surfaces: {path.name}"
        # Every non-accent opaque pixel should stay neutral, not blue or brown.
        accent = rgb[spread >= 0.025]
        if stem in ("1-soft-s", "4-pulse"):
            correct_hue = (accent[:, 2] >= accent[:, 1]) & (accent[:, 1] > accent[:, 0])
        elif stem == "3-fader":
            correct_hue = (accent[:, 1] > accent[:, 0]) & (accent[:, 0] >= accent[:, 2])
        else:
            correct_hue = (accent[:, 0] > accent[:, 1]) & (accent[:, 1] >= accent[:, 2])
        assert np.all(correct_hue), f"Unexpected accent hue: {path.name}"
        contrast = {}
        for label, background in (("dark", 15 / 255), ("white", 1.0)):
            delta = np.max(np.abs(rgb - background), axis=1)
            contrast[label] = round(float((delta > 0.10).mean()), 4)
            assert contrast[label] > 0.80, f"Low visibility on {label}: {path.name}"
        results.append({"file": path.name, "size": [size, size], "corner_alpha": 0,
                        "opaque_pixels": count, "bounds": bounds, "margins": margins,
                        "visible_fraction": contrast})
        bpy.data.images.remove(img)

report = ROOT / ".checks" / "pixels.json"
report.parent.mkdir(exist_ok=True)
report.write_text(json.dumps(results, indent=2) + "\n")
for result in results:
    print(json.dumps(result))
print(f"PASS: {len(results)} PNGs; size, transparency, content, margins, centering, palette, and contrast.")
