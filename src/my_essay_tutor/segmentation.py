"""Classical (non-neural) line segmentation.

TrOCR is a *line-level* recognizer: it was trained on single cropped text
lines, not full pages, so feeding it a whole essay photo directly degrades
fast once there is more than one line. A proper neural segmenter (this is
what Kraken specializes in) would fix that, but it needs its own pytorch
model resident in memory -- exactly the kind of local ML load that already
OOM-killed the DSG Compliance service on Render's 512MB free tier. Since
this app targets the same free tier, we use a lightweight horizontal
ink-density projection instead: no model, no extra memory, good enough to
split neatly-written student essays into line strips before each strip is
sent to TrOCR. Revisit with Kraken's segmenter if this ever runs on a
worker with real memory headroom.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def segment_lines(image: Image.Image, min_line_height: int = 10, pad: int = 4) -> list[Image.Image]:
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)

    threshold = arr.mean() - 0.5 * arr.std()
    ink_mask = arr < threshold
    row_density = ink_mask.sum(axis=1)

    if row_density.max() == 0:
        return [image]

    is_text_row = row_density > max(1.0, row_density.max() * 0.02)

    lines: list[Image.Image] = []
    start: int | None = None
    height = arr.shape[0]

    for y in range(height):
        if is_text_row[y] and start is None:
            start = y
        elif not is_text_row[y] and start is not None:
            top, bottom = max(0, start - pad), min(height, y + pad)
            if bottom - top >= min_line_height:
                lines.append(image.crop((0, top, image.width, bottom)))
            start = None

    if start is not None:
        top = max(0, start - pad)
        if height - top >= min_line_height:
            lines.append(image.crop((0, top, image.width, height)))

    return lines if lines else [image]
