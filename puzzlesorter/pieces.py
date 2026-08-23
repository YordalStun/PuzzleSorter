"""Detect individual loose puzzle pieces in the working photo.

Pieces are found as high-texture blobs outside the assembled board region and
outside any user-specified clutter exclusion boxes, then piles of touching
pieces are split into individual pieces via watershed, seeded from an
expected single-piece pixel area.
"""
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

Rect = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class Piece:
    id: int
    mask: np.ndarray  # full-photo-size uint8 0/255 mask of just this piece
    bbox: Rect
    centroid: Tuple[float, float]
    angle_hint: float  # minAreaRect angle, degrees; a coarse orientation prior


def build_roi_mask(photo_shape, board_mask=None, exclude_rects: Sequence[Rect] = (),
                    margin_top=0, margin_bottom=0, margin_left=0, margin_right=0,
                    board_dilate=31):
    h, w = photo_shape[:2]
    roi = np.full((h, w), 255, np.uint8)
    if margin_top:
        roi[:margin_top, :] = 0
    if margin_bottom:
        roi[h - margin_bottom:, :] = 0
    if margin_left:
        roi[:, :margin_left] = 0
    if margin_right:
        roi[:, w - margin_right:] = 0
    for (x, y, rw, rh) in exclude_rects:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + rw), min(h, y + rh)
        roi[y0:y1, x0:x1] = 0
    if board_mask is not None:
        dilated = cv2.dilate(board_mask, np.ones((board_dilate, board_dilate), np.uint8))
        roi[dilated > 0] = 0
    return roi


def _foreground_mask(photo_bgr, roi_mask, blur_ksize=9, close_ksize=7, open_ksize=3):
    gray = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    energy = cv2.boxFilter(np.abs(lap), -1, (blur_ksize, blur_ksize))
    energy_norm = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(energy_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask[roi_mask == 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_ksize, close_ksize), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8))
    return mask


def _split_blob(blob_mask, expected_area, min_piece_area):
    """Split a (possibly multi-piece) blob mask into per-piece masks via watershed."""
    area = int(np.count_nonzero(blob_mask))
    n_est = max(1, round(area / expected_area))
    if n_est <= 1:
        return [blob_mask]

    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    min_dist = max(3, int(0.30 * np.sqrt(expected_area)))
    coords = peak_local_max(dist, min_distance=min_dist, labels=blob_mask,
                             num_peaks=max(n_est * 2, n_est + 2))
    if len(coords) == 0:
        return [blob_mask]

    peak_mask = np.zeros(dist.shape, bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)

    blob3 = cv2.cvtColor(blob_mask, cv2.COLOR_GRAY2BGR)
    ws_markers = (markers + 1).astype(np.int32)
    ws_markers[blob_mask == 0] = 0
    cv2.watershed(blob3, ws_markers)

    pieces = []
    for label in range(2, markers.max() + 2):
        piece_mask = np.zeros(blob_mask.shape, np.uint8)
        piece_mask[ws_markers == label] = 255
        if np.count_nonzero(piece_mask) >= min_piece_area:
            pieces.append(piece_mask)
    return pieces if pieces else [blob_mask]


def detect_pieces(photo_bgr, roi_mask, expected_piece_area,
                   min_area_factor=0.30, max_single_factor=1.6,
                   max_cluster_factor=10.0, max_aspect_ratio=3.0) -> List[Piece]:
    """Find loose puzzle pieces within roi_mask.

    expected_piece_area: approximate pixel area of a single piece in this
    photo (see align.local_scale to convert a target-space estimate here).
    """
    fg = _foreground_mask(photo_bgr, roi_mask)
    min_area = expected_piece_area * min_area_factor

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pieces: List[Piece] = []
    next_id = 1
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if area > expected_piece_area * max_cluster_factor:
            # implausibly large for a piece pile (likely leftover clutter) - skip
            continue

        blob_mask = np.zeros(fg.shape, np.uint8)
        cv2.drawContours(blob_mask, [c], -1, 255, cv2.FILLED)

        if area <= expected_piece_area * max_single_factor:
            sub_masks = [blob_mask]
        else:
            sub_masks = _split_blob(blob_mask, expected_piece_area, min_area)

        for pm in sub_masks:
            a = np.count_nonzero(pm)
            if a < min_area:
                continue
            ys, xs = np.nonzero(pm)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            w, h = x1 - x0 + 1, y1 - y0 + 1
            if max(w, h) / max(1, min(w, h)) > max_aspect_ratio:
                # a real piece (or small cluster of them) is roughly blob-shaped;
                # a thin sliver is almost always an edge/shadow/wire artifact
                continue
            M = cv2.moments(pm, binaryImage=True)
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

            sub_contours, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            angle = 0.0
            if sub_contours:
                biggest = max(sub_contours, key=cv2.contourArea)
                if len(biggest) >= 5:
                    angle = cv2.minAreaRect(biggest)[-1]

            pieces.append(Piece(
                id=next_id,
                mask=pm,
                bbox=(int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)),
                centroid=(float(cx), float(cy)),
                angle_hint=float(angle),
            ))
            next_id += 1

    return pieces
