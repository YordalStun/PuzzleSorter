"""Match each detected loose piece to a location + rotation in the target image."""
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .align import Alignment, local_affine


@dataclass
class Match:
    piece_id: int
    target_xy: Tuple[float, float]
    photo_xy: Tuple[float, float]
    rotation_degrees: float  # rotation to apply to the piece, in the photo frame
    score: float


def _rotate_with_mask(bgr, mask, angle):
    h, w = mask.shape[:2]
    diag = int(np.ceil(np.hypot(h, w))) + 2
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += diag / 2 - cx
    M[1, 2] += diag / 2 - cy
    rot_img = cv2.warpAffine(bgr, M, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rot_mask = cv2.warpAffine(mask, M, (diag, diag), flags=cv2.INTER_NEAREST, borderValue=0)
    ys, xs = np.nonzero(rot_mask)
    if len(xs) == 0:
        return rot_img, rot_mask
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return rot_img[y0:y1 + 1, x0:x1 + 1], rot_mask[y0:y1 + 1, x0:x1 + 1]


def build_valid_mask(target_shape, polygon, inset_px=12):
    """A 0/255 mask, sized like the target image, that is 255 only inside
    `polygon` (Nx2 points), inset by a small FIXED number of pixels. Used to
    keep matches from landing outside the picture - e.g. in a tilted quad's
    axis-aligned bounding rect, the corners between the quad and its bounding
    box fall outside the actual picture (a box's cardboard border, glare, a
    hand holding it, etc.) and can otherwise win spurious high-confidence
    matches against pieces that happen to be similarly low-detail.

    inset_px is a fixed pixel margin rather than a percentage: real jigsaw
    puzzles have genuine pieces right up to the picture's edge (border
    pieces), so shrinking proportionally to the polygon's size (as an earlier
    version of this function did) throws away a lot of real, matchable area
    for a big picture - all that's actually needed is a few pixels of
    tolerance for detection/alignment imprecision at the true edge.
    """
    polygon = np.asarray(polygon, dtype=np.int32)
    mask = np.zeros(target_shape[:2], np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    if inset_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset_px + 1, 2 * inset_px + 1))
        mask = cv2.erode(mask, kernel)
    return mask


def _match_at_angle(target_bgr, piece_bgr, piece_mask, angle, valid_mask=None):
    rot_img, rot_mask = _rotate_with_mask(piece_bgr, piece_mask, angle)
    th, tw = rot_mask.shape[:2]
    if th >= target_bgr.shape[0] or tw >= target_bgr.shape[1] or th < 4 or tw < 4:
        return -1.0, None, None
    mask3 = cv2.merge([rot_mask, rot_mask, rot_mask])
    res = cv2.matchTemplate(target_bgr, rot_img, cv2.TM_CCOEFF_NORMED, mask=mask3)
    res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    if valid_mask is not None:
        # a candidate top-left position is only valid if the WHOLE template
        # footprint placed there stays inside valid_mask; erode (top-left
        # anchored, so it matches matchTemplate's top-left indexing) by the
        # template size to get exactly that per-position test.
        kernel = np.ones((th, tw), np.uint8)
        eroded = cv2.erode(valid_mask, kernel, anchor=(0, 0))
        valid_res = eroded[:res.shape[0], :res.shape[1]]
        res = np.where(valid_res > 0, res, -1.0)
    _, maxval, _, maxloc = cv2.minMaxLoc(res)
    return float(maxval), maxloc, (tw, th)


def match_piece_to_target(piece_bgr, piece_mask, target_bgr, search_rect,
                           scale_photo_per_target,
                           coarse_step=12, fine_step=2, fine_range=14,
                           coarse_downscale=0.35, fine_window_factor=2.5,
                           top_k_candidates=5, valid_mask=None):
    """Search rotation (0-360) and position within target_bgr[search_rect] for the
    best match of a piece crop. Returns (angle_deg, score, (target_x, target_y), size).

    Two-stage coarse-to-fine search for speed: the full search_rect is searched at
    reduced resolution over all coarse angles first; the best hit then seeds a small
    full-resolution window searched over fine angle steps around the coarse angle.

    valid_mask: optional 0/255 mask, same size as target_bgr, restricting where a
    match may land (see build_valid_mask). search_rect is typically a generous
    axis-aligned bounding box, so this is what actually keeps matches inside the
    true (possibly tilted) picture area.

    angle_deg is the rotation applied to piece_bgr (cv2 convention, positive =
    counter-clockwise) to align it with the target at the returned location.
    (target_x, target_y) is the top-left corner of the matched patch within the
    FULL target image (search_rect offset already applied).
    """
    sx, sy, sw, sh = search_rect
    target_crop = target_bgr[sy:sy + sh, sx:sx + sw]
    valid_crop = valid_mask[sy:sy + sh, sx:sx + sw] if valid_mask is not None else None

    inv_scale = 1.0 / scale_photo_per_target
    rw = max(4, int(round(piece_bgr.shape[1] * inv_scale)))
    rh = max(4, int(round(piece_bgr.shape[0] * inv_scale)))
    piece_resized = cv2.resize(piece_bgr, (rw, rh), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(piece_mask, (rw, rh), interpolation=cv2.INTER_NEAREST)

    # --- coarse stage: full search_rect, downscaled ---
    small_target = cv2.resize(target_crop, None, fx=coarse_downscale, fy=coarse_downscale,
                               interpolation=cv2.INTER_AREA)
    small_piece = cv2.resize(piece_resized, None, fx=coarse_downscale, fy=coarse_downscale,
                              interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask_resized, None, fx=coarse_downscale, fy=coarse_downscale,
                             interpolation=cv2.INTER_NEAREST)
    if small_piece.shape[0] < 4 or small_piece.shape[1] < 4:
        small_target, small_piece, small_mask = target_crop, piece_resized, mask_resized
        small_valid = valid_crop
        eff_downscale = 1.0
    else:
        small_valid = (cv2.resize(valid_crop, (small_target.shape[1], small_target.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                        if valid_crop is not None else None)
        eff_downscale = coarse_downscale

    # every coarse angle's best hit is a candidate; the true global optimum can fall
    # in a different angle bucket than the single best coarse score once downscaling
    # blurs detail, so refine the top-K distinct candidates rather than just one.
    coarse_hits = []  # (val, angle, loc)
    for angle in range(0, 360, coarse_step):
        val, loc, _size = _match_at_angle(small_target, small_piece, small_mask, angle,
                                           valid_mask=small_valid)
        if loc is not None:
            coarse_hits.append((val, angle, loc))

    if not coarse_hits:
        return None
    coarse_hits.sort(key=lambda t: -t[0])
    top_candidates = coarse_hits[:top_k_candidates]

    diag = int(np.ceil(np.hypot(*mask_resized.shape[:2]))) + 2
    win_pad = int(diag * fine_window_factor)

    best_val, best_loc, best_size, best_final_angle, best_offset = -1.0, None, None, 0.0, (0, 0)
    for _cval, cangle, cloc in top_candidates:
        full_x = cloc[0] / eff_downscale
        full_y = cloc[1] / eff_downscale
        wx0 = max(0, int(full_x - win_pad))
        wy0 = max(0, int(full_y - win_pad))
        wx1 = min(target_crop.shape[1], int(full_x + diag + win_pad))
        wy1 = min(target_crop.shape[0], int(full_y + diag + win_pad))
        window = target_crop[wy0:wy1, wx0:wx1]
        window_valid = valid_crop[wy0:wy1, wx0:wx1] if valid_crop is not None else None

        lo, hi = cangle - fine_range, cangle + fine_range + 1
        for angle in np.arange(lo, hi, fine_step):
            val, loc, size = _match_at_angle(window, piece_resized, mask_resized, angle,
                                              valid_mask=window_valid)
            if val > best_val:
                best_val, best_loc, best_size = val, loc, size
                best_final_angle, best_offset = angle, (wx0, wy0)

    if best_loc is None:
        return None
    target_x = sx + best_offset[0] + best_loc[0] + best_size[0] / 2.0
    target_y = sy + best_offset[1] + best_loc[1] + best_size[1] / 2.0
    return float(best_final_angle) % 360.0, best_val, (target_x, target_y), best_size


def match_all(pieces, photo_bgr, target_bgr, alignment: Alignment, search_rect,
              approx_target_point, **kwargs) -> List[Match]:
    """approx_target_point: a target-space (x, y) near the board (e.g. the picture's
    center) used to seed the photo<->target scale estimate before the exact match
    location is known. The homography's scale/rotation vary slowly across the table
    plane, so this is a fine approximation for pieces near the board."""
    H = alignment.homography
    seed_scale, _ = local_affine(alignment, approx_target_point)
    matches = []
    for piece in pieces:
        x, y, w, h = piece.bbox
        pad = 2
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(photo_bgr.shape[1], x + w + pad), min(photo_bgr.shape[0], y + h + pad)
        piece_bgr = photo_bgr[y0:y1, x0:x1]
        piece_mask = piece.mask[y0:y1, x0:x1]

        result = match_piece_to_target(piece_bgr, piece_mask, target_bgr, search_rect,
                                        scale_photo_per_target=seed_scale, **kwargs)
        if result is None:
            continue
        angle, score, (tx, ty), _ = result

        pt = cv2.perspectiveTransform(np.float32([[[tx, ty]]]), H)[0, 0]
        _, rot_h = local_affine(alignment, (tx, ty))
        rotation_photo = (angle + rot_h) % 360.0

        matches.append(Match(
            piece_id=piece.id,
            target_xy=(tx, ty),
            photo_xy=(float(pt[0]), float(pt[1])),
            rotation_degrees=rotation_photo,
            score=score,
        ))
    return matches


def flag_conflicts(matches: List[Match], min_distance) -> dict:
    """Greedy non-max suppression over target_xy: returns {piece_id: True/False}
    marking matches whose claimed target spot is within min_distance of a
    higher-scoring match's spot (likely two pieces competing for one location,
    usually because one of them is a mismatch)."""
    ranked = sorted(matches, key=lambda m: -m.score)
    accepted_xy = []
    conflicted = {}
    for m in ranked:
        tx, ty = m.target_xy
        is_conflict = any(np.hypot(tx - ax, ty - ay) < min_distance for ax, ay in accepted_xy)
        conflicted[m.piece_id] = is_conflict
        if not is_conflict:
            accepted_xy.append((tx, ty))
    return conflicted
