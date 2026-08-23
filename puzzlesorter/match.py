"""Match each detected loose piece to a location + rotation in the target image."""
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from . import color
from .align import Alignment, local_affine


@dataclass
class Match:
    piece_id: int
    target_xy: Tuple[float, float]
    photo_xy: Tuple[float, float]
    rotation_degrees: float  # rotation to apply to the piece, in the photo frame
    score: float  # combined score (structural match x color agreement) when
    # color_correction was supplied to match_all, otherwise equal to ncc_score
    ncc_score: float = None
    color_distance: float = None  # Lab distance at the matched location; None if unscored
    target_size: Tuple[float, float] = None  # matched footprint (w, h) in target-space pixels


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
        # cv2.erode's default border handling treats pixels outside the array
        # as foreground, specifically so shapes touching the array edge don't
        # erode there - the opposite of what's wanted here, since the polygon
        # can itself touch or nearly touch the target array's true boundary
        # (e.g. a box photo where the picture fills most of the frame), and
        # that is exactly where this margin matters most. borderValue=0 makes
        # erosion treat "off the array" the same as "off the polygon".
        mask = cv2.erode(mask, kernel, borderValue=0)
    return mask


def build_border_band_mask(target_shape, polygon, band_px):
    """A 0/255 mask: the polygon's own area minus everything more than
    band_px in from its boundary - i.e. a band hugging the picture's edge,
    roughly one outer row/column of grid cells wide. A piece with a detected
    straight edge (see shape.classify_piece_shape) is physically a border
    piece and can only belong somewhere in this band, regardless of what its
    printed content suggests."""
    full = build_valid_mask(target_shape, polygon, inset_px=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    # see build_valid_mask: borderValue=0 so erosion isn't a no-op wherever
    # `full` touches the array's true edge (band_px deep bands need to shrink
    # in from there too, not just from the polygon's own interior boundary)
    inset = cv2.erode(full, kernel, borderValue=0)
    return cv2.bitwise_and(full, cv2.bitwise_not(inset))


def build_corner_regions_mask(target_shape, polygon, radius_px):
    """A 0/255 mask: small disks at each of the polygon's 4 corners. A piece
    with 2+ detected straight edges is physically a corner piece and can
    only belong at one of these 4 spots."""
    mask = np.zeros(target_shape[:2], np.uint8)
    for corner in np.asarray(polygon, dtype=np.float64):
        cv2.circle(mask, tuple(corner.astype(int)), radius_px, 255, -1)
    return mask


def restrict_to_gaps(valid_mask_target, gap_mask_photo, homography, target_shape):
    """AND a target-space valid_mask with a photo-space gap mask (see
    board.find_gaps), so matches are only allowed to land where a piece could
    actually still go - not on top of one that's already placed.

    Points in the picture polygon that map to filled (non-gap) board area, or
    to nothing recognizable (a piece's true home not yet reached by the
    board-detection heuristic), are excluded rather than silently trusted.
    """
    Hinv = np.linalg.inv(homography)
    h, w = target_shape[:2]
    gap_in_target = cv2.warpPerspective(gap_mask_photo, Hinv, (w, h), flags=cv2.INTER_NEAREST)
    return cv2.bitwise_and(valid_mask_target, gap_in_target)


def _correlation_map(target_bgr, piece_bgr, piece_mask, angle, valid_mask=None):
    rot_img, rot_mask = _rotate_with_mask(piece_bgr, piece_mask, angle)
    th, tw = rot_mask.shape[:2]
    if th >= target_bgr.shape[0] or tw >= target_bgr.shape[1] or th < 4 or tw < 4:
        return None, (tw, th)
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
    return res, (tw, th)


def _match_at_angle(target_bgr, piece_bgr, piece_mask, angle, valid_mask=None):
    res, size = _correlation_map(target_bgr, piece_bgr, piece_mask, angle, valid_mask)
    if res is None:
        return -1.0, None, None
    _, maxval, _, maxloc = cv2.minMaxLoc(res)
    return float(maxval), maxloc, size


def _match_at_angle_multi(target_bgr, piece_bgr, piece_mask, angle, valid_mask=None, top_n=3):
    """Like _match_at_angle, but returns up to top_n well-separated local
    maxima instead of just the global one - two very different locations can
    both score well at the SAME rotation (no rotation difference at all
    between them), and colour-aware re-ranking downstream can only consider
    candidates that make it into this list in the first place."""
    res, size = _correlation_map(target_bgr, piece_bgr, piece_mask, angle, valid_mask)
    if res is None:
        return []
    tw, th = size
    min_dist = max(1, int(0.5 * min(tw, th)))
    work = res.copy()
    hits = []
    for _ in range(top_n):
        _, maxval, _, maxloc = cv2.minMaxLoc(work)
        if maxval <= -1.0:
            break
        hits.append((float(maxval), maxloc, size))
        x0, y0 = max(0, maxloc[0] - min_dist), max(0, maxloc[1] - min_dist)
        x1, y1 = min(work.shape[1], maxloc[0] + min_dist), min(work.shape[0], maxloc[1] + min_dist)
        work[y0:y1, x0:x1] = -1.0
    return hits


def match_piece_to_target(piece_bgr, piece_mask, target_bgr, search_rect,
                           scale_photo_per_target,
                           coarse_step=12, fine_step=2, fine_range=14,
                           coarse_downscale=0.35, fine_window_factor=0.15,
                           top_k_candidates=8, valid_mask=None, color_correction=None):
    """Search rotation (0-360) and position within target_bgr[search_rect] for the
    best match of a piece crop.
    Returns (angle_deg, ncc_score, combined_score, color_dist, (target_x, target_y), size).

    Two-stage coarse-to-fine search for speed: the full search_rect is searched at
    reduced resolution over all coarse angles first; the best hit then seeds a small
    full-resolution window searched over fine angle steps around the coarse angle.

    valid_mask: optional 0/255 mask, same size as target_bgr, restricting where a
    match may land (see build_valid_mask). search_rect is typically a generous
    axis-aligned bounding box, so this is what actually keeps matches inside the
    true (possibly tilted) picture area.

    color_correction: optional Lab offset (see color.estimate_color_correction).
    If given, the top-K candidates are re-ranked by ncc_score * color_agreement
    rather than ncc_score alone - matchTemplate's correlation rewards structural
    (edge/brightness-pattern) alignment and is surprisingly tolerant of outright
    hue mismatches, so without this a green patch can win against a blue one.

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

    # every coarse angle can contribute several spatially-distinct candidates
    # (two very different locations can score well at the SAME rotation - no
    # rotation difference between them at all), not just its single best hit,
    # and the true global optimum can also fall in a different angle bucket
    # than the single best coarse score once downscaling blurs detail. Refine
    # the top-K candidates pooled across all of that, not just one per angle.
    coarse_hits = []  # (val, angle, loc)
    for angle in range(0, 360, coarse_step):
        for val, loc, _size in _match_at_angle_multi(small_target, small_piece, small_mask,
                                                       angle, valid_mask=small_valid,
                                                       top_n=3):
            coarse_hits.append((val, angle, loc))

    if not coarse_hits:
        return None
    coarse_hits.sort(key=lambda t: -t[0])
    top_candidates = coarse_hits[:top_k_candidates]

    diag = int(np.ceil(np.hypot(*mask_resized.shape[:2]))) + 2
    # must stay well under the coarse stage's own candidate-separation distance
    # (~0.5x piece size - see _match_at_angle_multi's min_dist), or two
    # genuinely distinct candidates' fine-search windows overlap and both
    # collapse onto whichever one has the single strongest peak, defeating the
    # entire point of keeping candidates separate for colour re-ranking below.
    # This only needs to cover the coarse stage's own discretization error
    # (on the order of 1/coarse_downscale target pixels), not the piece size.
    win_pad = max(6, int(diag * fine_window_factor))

    # refine each top-K coarse candidate to its own local best, independently,
    # instead of merging into one running best - color re-ranking below needs
    # to compare distinct candidate locations against each other.
    refined = []  # (ncc_val, angle, loc, size, offset)
    for _cval, cangle, cloc in top_candidates:
        full_x = cloc[0] / eff_downscale
        full_y = cloc[1] / eff_downscale
        wx0 = max(0, int(full_x - win_pad))
        wy0 = max(0, int(full_y - win_pad))
        wx1 = min(target_crop.shape[1], int(full_x + diag + win_pad))
        wy1 = min(target_crop.shape[0], int(full_y + diag + win_pad))
        window = target_crop[wy0:wy1, wx0:wx1]
        window_valid = valid_crop[wy0:wy1, wx0:wx1] if valid_crop is not None else None

        c_best_val, c_best_loc, c_best_size, c_best_angle = -1.0, None, None, cangle
        lo, hi = cangle - fine_range, cangle + fine_range + 1
        for angle in np.arange(lo, hi, fine_step):
            val, loc, size = _match_at_angle(window, piece_resized, mask_resized, angle,
                                              valid_mask=window_valid)
            if val > c_best_val:
                c_best_val, c_best_loc, c_best_size, c_best_angle = val, loc, size, angle
        if c_best_loc is not None:
            refined.append((c_best_val, c_best_angle, c_best_loc, c_best_size, (wx0, wy0)))

    if not refined:
        return None

    if color_correction is not None:
        piece_lab = color.mean_lab(piece_resized, mask_resized)
        scored = []
        for ncc_val, angle, loc, size, offset in refined:
            rot_img, rot_mask = _rotate_with_mask(piece_resized, mask_resized, angle)
            tx0, ty0 = offset[0] + loc[0], offset[1] + loc[1]
            patch = target_crop[ty0:ty0 + size[1], tx0:tx0 + size[0]]
            patch_lab = color.mean_lab(patch, rot_mask[:patch.shape[0], :patch.shape[1]])
            if patch_lab is None:
                agreement, dist = 1.0, 0.0
            else:
                agreement, dist = color.color_agreement(piece_lab, patch_lab, color_correction)
            combined = ncc_val * agreement
            scored.append((combined, ncc_val, dist, angle, loc, size, offset))
        scored.sort(key=lambda t: -t[0])
        combined_val, ncc_val, color_dist, best_final_angle, best_loc, best_size, best_offset = scored[0]
    else:
        refined.sort(key=lambda t: -t[0])
        ncc_val, best_final_angle, best_loc, best_size, best_offset = refined[0]
        combined_val, color_dist = ncc_val, None

    target_x = sx + best_offset[0] + best_loc[0] + best_size[0] / 2.0
    target_y = sy + best_offset[1] + best_loc[1] + best_size[1] / 2.0
    return (float(best_final_angle) % 360.0, ncc_val, combined_val, color_dist,
            (target_x, target_y), best_size)


def match_all(pieces, photo_bgr, target_bgr, alignment: Alignment, search_rect,
              approx_target_point, valid_mask=None, border_mask=None, corner_mask=None,
              **kwargs) -> List[Match]:
    """approx_target_point: a target-space (x, y) near the board (e.g. the picture's
    center) used to seed the photo<->target scale estimate before the exact match
    location is known. The homography's scale/rotation vary slowly across the table
    plane, so this is a fine approximation for pieces near the board.

    border_mask/corner_mask (see build_border_band_mask/build_corner_regions_mask):
    if given, a piece whose segmented silhouette has a detected straight edge
    is physically constrained to the border band, and one with 2+ straight
    edges to a corner - a real geometric fact independent of what its printed
    content matches, so it's applied as a hard AND on top of valid_mask
    rather than left to content-matching confidence alone.

    That constraint can still legitimately come up empty for a specific
    piece - e.g. a larger-than-average piece whose full bounding box doesn't
    fit inside the remaining (already fragmented by other placements) gap
    area within the band, even though the band is generously sized on
    average (verified against a real photo: this isn't a rare corner case,
    it affected roughly half of that run's real border pieces). Rather than
    silently dropping the piece from the results, fall back to searching
    under the plain valid_mask so it still gets a match - just without the
    geometric backup.
    """
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

        piece_valid_mask = valid_mask
        if piece.straight_edge_count >= 2 and corner_mask is not None:
            piece_valid_mask = cv2.bitwise_and(valid_mask, corner_mask)
        elif piece.straight_edge_count == 1 and border_mask is not None:
            piece_valid_mask = cv2.bitwise_and(valid_mask, border_mask)

        result = match_piece_to_target(piece_bgr, piece_mask, target_bgr, search_rect,
                                        scale_photo_per_target=seed_scale,
                                        valid_mask=piece_valid_mask, **kwargs)
        if result is None and piece_valid_mask is not valid_mask:
            result = match_piece_to_target(piece_bgr, piece_mask, target_bgr, search_rect,
                                            scale_photo_per_target=seed_scale,
                                            valid_mask=valid_mask, **kwargs)
        if result is None:
            continue
        angle, ncc_score, combined_score, color_dist, (tx, ty), size = result

        pt = cv2.perspectiveTransform(np.float32([[[tx, ty]]]), H)[0, 0]
        _, rot_h = local_affine(alignment, (tx, ty))
        rotation_photo = (angle + rot_h) % 360.0

        matches.append(Match(
            piece_id=piece.id,
            target_xy=(tx, ty),
            photo_xy=(float(pt[0]), float(pt[1])),
            rotation_degrees=rotation_photo,
            score=combined_score,
            ncc_score=ncc_score,
            color_distance=color_dist,
            target_size=(float(size[0]), float(size[1])),
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
