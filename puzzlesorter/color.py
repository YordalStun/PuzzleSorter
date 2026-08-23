"""Color-consistency scoring for piece matches.

matchTemplate's normalized cross-correlation rewards structural alignment
(edges, the relative pattern of light/dark) but is surprisingly tolerant of
outright hue mismatches - two patches with similar edges can score well even
if one is green and the other blue. This adds an explicit color check on top,
calibrated against the part of the puzzle already known to be correct (the
assembled region), since the two source photos are rarely under identical
lighting/white balance.
"""
import cv2
import numpy as np


def estimate_color_correction(photo_bgr, warped_target_bgr, calibration_mask):
    """Mean Lab offset (photo - warped_target) over calibration_mask: how to
    shift a target-image color to look like it does under the photo's
    lighting. calibration_mask should cover only pixels known to be correctly
    filled (e.g. the assembled region, minus any gaps within it)."""
    photo_lab = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(warped_target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = calibration_mask > 0
    return photo_lab[m].mean(axis=0) - target_lab[m].mean(axis=0)


def mean_lab(bgr, mask=None):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is not None:
        m = mask > 0
        if not np.any(m):
            return None
        return lab[m].mean(axis=0)
    return lab.reshape(-1, 3).mean(axis=0)


def color_agreement(piece_lab, target_patch_lab, correction, scale=25.0):
    """(factor, distance): factor is 0-1, 1.0 = colors agree closely, falling
    off as they diverge. `scale` is the Lab distance (~perceptual color
    difference) at which agreement drops to ~37%; 25-30 is a clearly visible
    mismatch (e.g. green vs. blue), not just lighting/JPEG noise."""
    corrected = target_patch_lab + correction
    dist = float(np.linalg.norm(piece_lab - corrected))
    return float(np.exp(-dist / scale)), dist
