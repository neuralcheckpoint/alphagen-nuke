"""Turn the matte drawn in Nuke into SAM 2 prompts.

A mask given to SAM 2 as a mask is used as is on its frame (the 2.1 configs
pass mask inputs straight to the output), so a rough roto would be tracked
as rough. A rough shape therefore becomes a box around it plus a positive
click in the middle of each of its largest parts, and SAM 2 finds the edge
itself. An exact matte is passed as a mask.
"""

import numpy as np
from scipy import ndimage

MAX_POINTS = 3
THRESHOLD = 0.5


def shape_prompt(alpha):
    """Box and positive points for a rough shape, in plate pixels (x right, y down).

    Returns ``(box, points)``: box = [x0, y0, x1, y1], points = [[x, y], ...].
    Each point sits where its part is thickest, so it lands on the object and
    not on the edge the shape only roughly follows.
    """
    inside = np.asarray(alpha) > THRESHOLD
    if not inside.any():
        raise ValueError("the prompt is empty")
    rows = np.flatnonzero(inside.any(axis=1))
    columns = np.flatnonzero(inside.any(axis=0))
    box = [float(columns[0]), float(rows[0]), float(columns[-1] + 1), float(rows[-1] + 1)]

    labels, count = ndimage.label(inside)
    sizes = ndimage.sum(inside, labels, index=range(1, count + 1))
    points = []
    for label in (np.argsort(sizes)[::-1][:MAX_POINTS] + 1):
        part = labels == label
        # Pad so a part touching the frame edge still has a distance to it.
        depth = ndimage.distance_transform_edt(np.pad(part, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(np.argmax(depth), depth.shape)
        points.append([float(x) + 0.5, float(y) + 0.5])
    return box, points


def matte_prompt(alpha):
    """Boolean mask for an exact matte."""
    inside = np.asarray(alpha) > THRESHOLD
    if not inside.any():
        raise ValueError("the prompt is empty")
    return inside
