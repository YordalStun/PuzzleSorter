"""Locate the already-assembled puzzle region within the working photo.

The assembled region (and any dense photographic content in general) has much
higher local texture/gradient energy than a plain mat, tablecloth, or table
surface. We use that to segment it out as a rectangle, which lets the rest of
the pipeline (a) exclude it from loose-piece detection and (b) cross the
target<->photo homography to know exactly which target-image pixels are
already placed.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BoardRegion:
    quad: np.ndarray  # 4x2 corners (rotated rect) of the assembled region, in photo pixel coords
    mask: np.ndarray  # full-size binary mask (uint8, 0/255) of the assembled region's contour
    contour: np.ndarray


def _texture_energy(gray, blur_ksize=25):
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    energy = cv2.boxFilter(np.abs(lap), -1, (blur_ksize, blur_ksize))
    return cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def find_assembled_region(photo_bgr, open_ksize=41, close_ksize=25):
    """Find the largest high-texture blob in the photo, i.e. the assembled puzzle.

    Uses a fairly large morphological opening to break thin bridges to
    adjacent loose-piece piles that happen to touch the assembled region, so
    the returned contour/quad reflects just the assembled block itself.
    """
    gray = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)
    energy = _texture_energy(gray)

    _, mask = cv2.threshold(energy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_ksize, close_ksize), np.uint8))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8))

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No high-texture region found; cannot locate the assembled puzzle")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    biggest = contours[0]
    if len(contours) > 1 and cv2.contourArea(contours[1]) > 0.3 * cv2.contourArea(biggest):
        raise RuntimeError(
            "Could not confidently isolate a single assembled-puzzle region "
            "(multiple similarly-sized high-texture blobs found)"
        )

    rect = cv2.minAreaRect(biggest)
    box = cv2.boxPoints(rect)

    full_mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(full_mask, [biggest], -1, 255, cv2.FILLED)

    return BoardRegion(quad=box, mask=full_mask, contour=biggest)
