"""Detect individual loose puzzle pieces in the working photo.

Pieces are found as high-texture blobs outside the assembled board region and
outside any user-specified clutter exclusion boxes, then piles of touching
pieces are split into individual pieces via watershed, seeded from an
expected single-piece pixel area. Each found piece's rough mask is then
refined locally (see _refine_piece_mask) to correct for texture-energy
detection under-capturing a piece that mixes high- and low-texture content.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

from .shape import PieceShape, classify_piece_shape

Rect = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class Piece:
    id: int
    mask: np.ndarray  # full-photo-size uint8 0/255 mask of just this piece
    bbox: Rect
    centroid: Tuple[float, float]
    angle_hint: float  # minAreaRect angle, degrees; a coarse orientation prior
    edge_shape: Optional[PieceShape] = None  # None if classification failed
    straight_edge_count: int = 0  # 0 if edge_shape is None - treated as "interior, unknown"


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


def _refine_piece_mask(photo_bgr, rough_mask, expected_piece_area=None,
                        pad=70, border_ring_px=12, min_half=90):
    """Grow/correct one piece's rough mask (from _foreground_mask) to its
    true extent.

    _foreground_mask uses a single texture-energy threshold over the WHOLE
    photo, so a piece that mixes high- and low-texture content (e.g. a dark
    printed shape on an otherwise plain patch) only gets its highest-contrast
    internal edges captured - confirmed on a real photo, where a piece's
    mask was under half its true size and a virtual-placement render visibly
    showed part of the piece missing.

    Once we roughly know where a piece is (which we don't at the
    _foreground_mask stage - that function's job is finding pieces in the
    first place), better signals become available that only make sense
    locally: a LOCAL background color sample (a thin ring around this one
    piece, not one global color - real table lighting varies enough that a
    single global background estimate misses some pieces and swallows the
    mat around others), 2-means color clustering, and mean-shift smoothing
    followed by the same local-background color distance.

    Tried individually against 16 real, hand-picked pieces (including
    deliberately hard ones: very pale content barely different from the mat,
    two pieces touching, an oversized non-piece blob) each of the 3 has its
    own failure mode severe enough to occasionally balloon a piece's mask
    out into surrounding mat or a neighboring piece - one piece's own shadow
    can throw off its local color sample, another piece's low contrast can
    defeat clustering. Requiring all 3 to unanimously agree on a pixel
    trades a bit of recall for reliably avoiding that: across those 16
    pieces this combination never once produced a wild over-capture, where
    requiring only 2-of-3 agreement still did on two of them.

    Strong texture-energy (recomputed locally, same as _foreground_mask) is
    ORed in on top of that unanimous vote rather than required alongside it:
    a perfectly flat but distinctly-colored region - this function's whole
    reason for existing - has ZERO texture-energy by construction, so
    requiring it too would silently defeat the fix (caught by a synthetic
    test with a piece half solid color, half random texture). Texture-energy
    has never shown the same blow-up failure mode as the color-based
    signals, so trusting it alone is safe.
    """
    ys, xs = np.nonzero(rough_mask)
    if len(xs) == 0:
        return rough_mask
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    ccx, ccy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(min_half, int(max(bw, bh) / 2 + pad))

    H0, W0 = photo_bgr.shape[:2]
    cx0, cy0 = max(0, int(ccx - half)), max(0, int(ccy - half))
    cx1, cy1 = min(W0, int(ccx + half)), min(H0, int(ccy + half))
    crop = photo_bgr[cy0:cy1, cx0:cx1]
    H, W = crop.shape[:2]
    if H < 2 * border_ring_px + 4 or W < 2 * border_ring_px + 4:
        return rough_mask
    local_center = (ccx - cx0, ccy - cy0)

    border = np.zeros((H, W), np.uint8)
    border[:border_ring_px, :] = 1
    border[-border_ring_px:, :] = 1
    border[:, :border_ring_px] = 1
    border[:, -border_ring_px:] = 1

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = np.median(lab[border > 0], axis=0)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    energy = cv2.boxFilter(np.abs(lap), -1, (9, 9))
    energy_norm = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, texture_vote = cv2.threshold(energy_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    color_votes = np.zeros((H, W), np.int32)

    color_dist = np.linalg.norm(lab - bg_lab, axis=2)
    color_votes += (color_dist > 14).astype(np.int32)

    Z = lab.reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
    _, km_labels, _ = cv2.kmeans(Z, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    km_labels = km_labels.reshape(H, W)
    bg_cluster = np.bincount(km_labels[border > 0].ravel(), minlength=2).argmax()
    color_votes += (km_labels != bg_cluster).astype(np.int32)

    shifted = cv2.pyrMeanShiftFiltering(crop, sp=12, sr=25)
    shifted_lab = cv2.cvtColor(shifted, cv2.COLOR_BGR2LAB).astype(np.float32)
    ms_dist = np.linalg.norm(shifted_lab - bg_lab, axis=2)
    color_votes += (ms_dist > 12).astype(np.int32)

    def _nearest_component(raw01):
        m = (raw01 > 0).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return np.zeros((H, W), np.uint8)
        best_label, best_d = None, None
        for lbl in range(1, n):
            lcx, lcy = centroids[lbl]
            area = stats[lbl, cv2.CC_STAT_AREA]
            d = np.hypot(lcx - local_center[0], lcy - local_center[1])
            if area > 0.5 * H * W:
                d += 1e6  # a blob covering half the crop is never "the piece"
            if best_d is None or d < best_d:
                best_d, best_label = d, lbl
        out = np.zeros((H, W), np.uint8)
        out[labels == best_label] = 255
        return out

    # texture and the unanimous color vote are each independently reduced to
    # a single bounded, nearest-to-center blob BEFORE being combined - OR'ing
    # the raw per-pixel signals together first (tried and measured on real
    # photos) lets morphological closing bridge a texture fragment and an
    # unrelated color fragment elsewhere in the crop into one runaway blob,
    # since closing only sees "pixels that are on", not which signal turned
    # them on. Combining two already-selected, already-sane shapes avoids
    # that while still recovering a piece that only ONE of the two signals
    # can see (texture on an otherwise-plain piece; color on an otherwise
    # low-contrast piece).
    texture_component = _nearest_component(texture_vote)
    color_component = _nearest_component((color_votes >= 3).astype(np.uint8))
    refined_local = cv2.bitwise_or(texture_component, color_component)

    refined_area = np.count_nonzero(refined_local)
    rough_area = np.count_nonzero(rough_mask)
    if refined_area < 0.15 * rough_area or refined_area > 4.0 * rough_area:
        # collapsed to near-nothing (seen on very low-contrast pieces where
        # none of these signals can see the piece at all) or ballooned
        # implausibly - either way the rough mask, even if undersized, beats
        # trusting a refinement this far from where it started
        return rough_mask
    if expected_piece_area is not None and refined_area > 3.0 * expected_piece_area:
        # relative-to-rough-mask alone isn't enough: a rough mask that's
        # ALREADY a plausible piece size can still have its refinement
        # swallow a large nearby confounder (a shadow, dark clutter) that
        # legitimately agrees with all 3 color signals over a wide area -
        # measured on a real photo, where this passed the 4x-relative check
        # (rough was itself already too generous) but was 12x the actual
        # known single-piece size
        return rough_mask

    refined_full = np.zeros((H0, W0), np.uint8)
    refined_full[cy0:cy1, cx0:cx1] = refined_local
    return refined_full


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
            pm = _refine_piece_mask(photo_bgr, pm, expected_piece_area=expected_piece_area)
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

            # only classify edges for something that plausibly IS a single
            # piece - a still-merged multi-piece cluster's outline isn't any
            # one piece's tab/blank pattern, and would just be noise here
            edge_shape = None
            if a <= expected_piece_area * max_single_factor:
                try:
                    edge_shape = classify_piece_shape(
                        pm, expected_piece_size=float(np.sqrt(expected_piece_area)))
                except (ValueError, cv2.error):
                    pass

            pieces.append(Piece(
                id=next_id,
                mask=pm,
                bbox=(int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)),
                centroid=(float(cx), float(cy)),
                angle_hint=float(angle),
                edge_shape=edge_shape,
                straight_edge_count=edge_shape.straight_count if edge_shape else 0,
            ))
            next_id += 1

    return pieces
