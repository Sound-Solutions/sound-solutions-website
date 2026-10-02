"""Make signature PNGs and self-contained SVG proof sheets.

Run after render.py using Blender's Python (bundled NumPy):
  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python-exit-code 1 -P brand/explore/astra/make_previews.py
All output stays beside this script.
Proof sheets are square so Quick Look does not crop them. The small marks are
exactly 84 by 84 CSS pixels, sourced from the delivered 252px PNGs.
"""

import base64
from pathlib import Path
import struct
import zlib

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parent
VERSION = 3
OPTIONS = [
    (1, "soft-s", "Soft S", "One rounded S in brushed graphite with a cyan inlay."),
    (2, "cone", "Cone", "A black speaker cone with a rubber surround and orange center ring."),
    (3, "fader", "Fader", "A single black console fader with a green position line."),
    (4, "pulse", "Pulse", "Five rounded audio bars with a cyan center pulse."),
    (5, "ripple", "Ripple", "Two round sound waves around an orange source."),
]


def area_resize(pixels, size, axis):
    """Integrate actual source-pixel coverage with nonnegative weights.

    A sharp resize filter creates inverse-color ringing beside colored inlays.
    Area averaging keeps neutral faces neutral and cannot invent another hue.
    """
    data = np.moveaxis(pixels, axis, 0)
    source_size = data.shape[0]
    bounds = np.linspace(0, source_size, size + 1)
    indices = np.minimum(bounds.astype(int), source_size - 1)
    fraction = (bounds - indices).reshape((-1,) + (1,) * (data.ndim - 1))
    prefix = np.concatenate((np.zeros_like(data[:1]), np.cumsum(data, axis=0)), axis=0)
    integral = prefix[indices] + data[indices] * fraction
    result = np.diff(integral, axis=0) / (source_size / size)
    return np.moveaxis(result, 0, axis)


def signature(source, target):
    image = bpy.data.images.load(str(source), check_existing=False)
    image.colorspace_settings.name = "Non-Color"
    width, height = image.size
    pixels = np.array(image.pixels[:], dtype=np.float64).reshape(height, width, 4)
    pixels[:, :, :3] *= pixels[:, :, 3:4]
    pixels = area_resize(area_resize(pixels, 252, 0), 252, 1)
    np.divide(pixels[:, :, :3], pixels[:, :, 3:4], out=pixels[:, :, :3], where=pixels[:, :, 3:4] > 0)
    # Blender exposes the bottom row first; PNG scanlines start at the top.
    rgba = np.clip(np.rint(pixels[::-1] * 255), 0, 255).astype(np.uint8)
    rgba[rgba[:, :, 3] == 0] = 0

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)

    encoded = b"\x89PNG\r\n\x1a\n"
    encoded += chunk(b"IHDR", struct.pack(">IIBBBBB", 252, 252, 8, 6, 0, 0, 0))
    encoded += chunk(b"sRGB", b"\x00")
    encoded += chunk(b"IDAT", zlib.compress(b"".join(b"\x00" + row.tobytes() for row in rgba), 9))
    encoded += chunk(b"IEND", b"")
    target.write_bytes(encoded)
    bpy.data.images.remove(image)


def embedded(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def image(path, x, y, size):
    return f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="{embedded(path)}"/>'


def label(text, x, y, size=16, color="#EDEDED"):
    return f'<text x="{x}" y="{y}" font-family="Helvetica,Arial,sans-serif" font-size="{size}" fill="{color}">{text}</text>'


def svg(size, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<rect width="{size}" height="{size}" fill="#0F0F0F"/>' + body + '</svg>')


overview = label(f"Sound Solutions / Logo exploration v{VERSION}", 46, 58, 28)
for index, slug, title, description in OPTIONS:
    large = ROOT / f"{index}-{slug}.png"
    small = ROOT / f"{index}-{slug}-sig.png"
    signature(large, small)
    # A large proof plus exactly 84px copies on each email background.
    proof = label(f"{index}. {title} / v{VERSION}", 24, 34, 20)
    proof += image(large, 112, 47, 288)
    proof += '<rect x="272" y="374" width="124" height="124" fill="#FFFFFF"/>'
    proof += image(small, 134, 394, 84)
    proof += image(small, 292, 394, 84)
    proof += label("84px / dark", 134, 367, 12, "#8A8A8A")
    proof += label("84px / white", 292, 367, 12, "#8A8A8A")
    (ROOT / f"{index}-{slug}-proof-v{VERSION}.svg").write_text(svg(512, proof))

    col, row = (index - 1) % 3, (index - 1) // 3
    x, y = 46 + col * 488, 110 + row * 654
    overview += label(f"{index}. {title}", x, y + 22, 23)
    overview += image(large, x + 26, y + 44, 386)
    overview += label("84px / dark", x + 86, y + 464, 14, "#8A8A8A")
    overview += label("84px / white", x + 246, y + 464, 14, "#8A8A8A")
    overview += f'<rect x="{x + 226}" y="{y + 481}" width="124" height="124" fill="#FFFFFF"/>'
    overview += image(small, x + 86, y + 501, 84)
    overview += image(small, x + 246, y + 501, 84)

(ROOT / f"review-v{VERSION}.svg").write_text(svg(1500, overview))
(ROOT / "NOTES.md").write_text("\n".join(f"{number}. **{title}** — {description}" for number, slug, title, description in OPTIONS) + "\n")
print("Created 5 signature PNGs, 5 individual proof sheets, overview, and NOTES.md.")
