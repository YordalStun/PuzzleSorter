"""Classify a puzzle piece's silhouette by edge type: straight, tab (bump
sticking out), or blank (notch cut in). A straight edge is only possible on
the picture's actual border, and two adjacent straight edges only on a
corner - a real geometric constraint on where a piece can go, independent of
what its printed content looks like.
"""
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class EdgeShape:
    kind: str  # "straight", "tab", "blank", or "unknown" (too little contour data)
    deviation: float  # perpendicular distance (px) driving the classification


@dataclass
class PieceShape:
    edges: List[EdgeShape]  # 4 entries, in minAreaRect corner order
    box: np.ndarray  # the 4 corners (float32) edges[i] runs from box[i] to box[(i+1)%4]

    @property
    def straight_count(self):
        return sum(1 for e in self.edges if e.kind == "straight")

    @property
    def is_border(self):
        return self.straight_count == 1

    @property
    def is_corner(self):
        return self.straight_count >= 2


def classify_piece_shape(mask, expected_piece_size, straight_tol_frac=0.09,
                          corner_margin_frac=0.12, smooth_frac=0.01,
                          tab_depth_frac=0.20) -> PieceShape:
    """mask: a piece's binary silhouette (any size canvas; only its own
    contour matters). expected_piece_size: the puzzle's typical single-piece
    side length in this mask's pixel units (e.g. sqrt of the expected piece
    area already used elsewhere in the pipeline) - deliberately an *external*
    estimate, not derived from this piece's own contour: a tab sticking out
    inflates minAreaRect, so bootstrapping the tab-removal kernel from the
    piece's own (already tab-distorted) bounding box undersized or oversized
    it depending on that piece's specific shape, which then mis-shapes the
    reference rectangle asymmetrically instead of consistently.

    Real segmented pieces have some boundary jitter from JPEG noise and
    thresholding, so the contour is lightly smoothed first - otherwise that
    jitter alone can look like a shallow tab on every side.

    minAreaRect of the RAW contour is a poor reference for the piece's core
    square: a tab sticking out on even one side drags the whole fitted
    rectangle bigger and off-axis, throwing off every other side's
    measurement too. Opening the mask first (erode then dilate) by roughly a
    tab's depth removes those protrusions while leaving the core body
    essentially unchanged - blanks don't need the same treatment, since a
    concave notch doesn't affect minAreaRect (which only responds to the
    outer/convex extent) - so this gives a clean reference rectangle to
    measure the *original* (un-opened) contour's deviation against.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("mask has no contour")
    contour = max(contours, key=cv2.contourArea)
    tab_depth_px = max(3, int(tab_depth_frac * expected_piece_size))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tab_depth_px + 1,) * 2)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    opened_contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not opened_contours:
        raise ValueError("piece shrank to nothing after opening - tab_depth_frac too large?")
    reference_contour = max(opened_contours, key=cv2.contourArea)

    # opening with a round kernel reliably gives a good CENTER and
    # ORIENTATION, but rounds the true corners off - the fitted rectangle's
    # own w/h is systematically smaller than the real piece by an amount
    # that depends on how much of each tab happened to survive erosion, not
    # a fixed offset. Trust the opened shape only for center/angle, and
    # impose the externally-known nominal size directly - real jigsaw grid
    # cells are close enough to square that a single size for both axes is
    # a reasonable approximation.
    (cx, cy), _wh, angle = cv2.minAreaRect(reference_contour)
    rect = ((cx, cy), (expected_piece_size, expected_piece_size), angle)
    box = cv2.boxPoints(rect)
    center = np.array(rect[0])

    perim = cv2.arcLength(contour, True)
    epsilon = max(1.0, smooth_frac * perim)
    smoothed = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float64)

    tol = straight_tol_frac * expected_piece_size
    # a point can project within an edge's along-edge span while actually
    # belonging to a different side entirely (e.g. a tab on the opposite
    # edge, or an adjacent edge's tab near a corner) - its perpendicular
    # distance would then reflect roughly the piece's own size, not a real
    # tab/blank depth, so also require plausible closeness to the edge line.
    max_plausible_dev = 1.4 * tab_depth_px

    edges = []
    for i in range(4):
        p0, p1 = box[i].astype(np.float64), box[(i + 1) % 4].astype(np.float64)
        edge_vec = p1 - p0
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-6:
            edges.append(EdgeShape("unknown", 0.0))
            continue
        edge_dir = edge_vec / edge_len
        normal = np.array([-edge_dir[1], edge_dir[0]])
        mid = (p0 + p1) / 2
        if np.dot(normal, mid - center) < 0:
            normal = -normal

        rel = smoothed - p0
        t = rel @ edge_dir
        d = rel @ normal
        in_span = ((t > corner_margin_frac * edge_len) & (t < (1 - corner_margin_frac) * edge_len)
                   & (np.abs(d) < max_plausible_dev))
        if not np.any(in_span):
            # no simplified-contour vertex anywhere near this edge at all -
            # approxPolyDP only keeps vertices where the contour deviates
            # from a straight line, so having none here is itself the
            # signal that this side is straight, not a lack of data.
            edges.append(EdgeShape("straight", 0.0))
            continue

        max_out = float(max(0.0, d[in_span].max()))
        max_in = float(max(0.0, -d[in_span].min()))
        deviation = max(max_out, max_in)
        if deviation < tol:
            edges.append(EdgeShape("straight", deviation))
        elif max_out >= max_in:
            edges.append(EdgeShape("tab", max_out))
        else:
            edges.append(EdgeShape("blank", max_in))

    return PieceShape(edges=edges, box=box)
