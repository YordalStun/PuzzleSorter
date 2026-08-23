"""Align a target reference image (e.g. a puzzle box photo) onto a photo
containing the assembled/partially-assembled puzzle, via feature matching.

Both photos can be taken at arbitrary rotation/angle/perspective — we don't
need either one to be a clean top-down scan. As long as the puzzle's picture
content is visible and roughly planar in both, ORB feature matching + RANSAC
recovers a homography between them.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Alignment:
    homography: np.ndarray  # maps target image pixel coords -> photo pixel coords
    inlier_matches: int
    total_matches: int
    target_quad_in_photo: np.ndarray  # 4x2 corners of the target image projected into the photo


def _detect_and_describe(gray, n_features=8000):
    orb = cv2.ORB_create(nfeatures=n_features, scoreType=cv2.ORB_HARRIS_SCORE)
    kp, desc = orb.detectAndCompute(gray, None)
    return kp, desc


def align_target_to_photo(target_bgr, photo_bgr, ratio_thresh=0.75, min_inliers=15):
    """Find a homography mapping target_bgr pixel coords to photo_bgr pixel coords.

    Raises RuntimeError if not enough reliable correspondence is found.
    """
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    photo_gray = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)

    kp1, desc1 = _detect_and_describe(target_gray)
    kp2, desc2 = _detect_and_describe(photo_gray)
    if desc1 is None or desc2 is None or len(kp1) < 2 or len(kp2) < 2:
        raise RuntimeError("Not enough texture to detect features in target or photo image")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)

    good = [m for m, n in raw_matches if m.distance < ratio_thresh * n.distance]

    if len(good) < min_inliers:
        raise RuntimeError(
            f"Only found {len(good)} candidate feature matches between target and photo "
            f"(need >= {min_inliers}). The images may not share enough visible content."
        )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    if H is None:
        raise RuntimeError("Homography estimation failed (RANSAC found no consistent model)")

    inliers = int(mask.sum())
    if inliers < min_inliers:
        raise RuntimeError(
            f"Only {inliers} inlier matches after RANSAC (need >= {min_inliers}); "
            f"alignment is unreliable."
        )

    h, w = target_gray.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    return Alignment(homography=H, inlier_matches=inliers, total_matches=len(good),
                      target_quad_in_photo=quad)


def warp_target_into_photo(target_bgr, photo_bgr, alignment: Alignment):
    """Warp the target image into the photo's coordinate frame (for debug/overlay)."""
    h, w = photo_bgr.shape[:2]
    return cv2.warpPerspective(target_bgr, alignment.homography, (w, h))


def local_jacobian(alignment: Alignment, at_xy):
    """Jacobian (2x2) of the homography's projective map at a target-space point,
    i.e. the best local linear approximation of target-pixel -> photo-pixel."""
    H = alignment.homography
    x, y = at_xy
    denom = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    dudx = (H[0, 0] * denom - (H[0, 0] * x + H[0, 1] * y + H[0, 2]) * H[2, 0]) / denom ** 2
    dudy = (H[0, 1] * denom - (H[0, 0] * x + H[0, 1] * y + H[0, 2]) * H[2, 1]) / denom ** 2
    dvdx = (H[1, 0] * denom - (H[1, 0] * x + H[1, 1] * y + H[1, 2]) * H[2, 0]) / denom ** 2
    dvdy = (H[1, 1] * denom - (H[1, 0] * x + H[1, 1] * y + H[1, 2]) * H[2, 1]) / denom ** 2
    return np.array([[dudx, dudy], [dvdx, dvdy]])


def local_scale(alignment: Alignment, at_xy):
    """Approximate local photo-pixels-per-target-pixel scale factor of H at a point."""
    J = local_jacobian(alignment, at_xy)
    svals = np.linalg.svd(J, compute_uv=False)
    return float(np.mean(svals))


def local_affine(alignment: Alignment, at_xy):
    """Decompose H's local Jacobian at a target-space point into (scale, rotation_degrees):
    the isotropic scale and pure-rotation angle of the closest rigid approximation of the
    target-pixel -> photo-pixel map there. rotation_degrees is measured the same way
    cv2.minAreaRect angles are (image y-axis pointing down).
    """
    J = local_jacobian(alignment, at_xy)
    U, S, Vt = np.linalg.svd(J)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        # reflection - shouldn't happen for a valid camera homography, but guard anyway
        U[:, -1] *= -1
        R = U @ Vt
    scale = float(np.mean(S))
    rotation_degrees = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return scale, rotation_degrees
