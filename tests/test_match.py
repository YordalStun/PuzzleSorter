import numpy as np
import cv2
import pytest

from puzzlesorter.match import (match_piece_to_target, match_all, _rotate_with_mask,
                                 build_valid_mask, build_border_band_mask,
                                 build_corner_regions_mask, restrict_to_gaps)
from puzzlesorter.pieces import Piece


def _synthetic_textured_image(size=900, seed=0):
    """A busy, non-repeating image so template matching has something distinctive
    to lock onto (unlike a blank or periodic pattern)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    for _ in range(400):
        pt1 = tuple(rng.integers(0, size, size=2).tolist())
        pt2 = tuple((np.array(pt1) + rng.integers(-80, 80, size=2)).tolist())
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        thickness = int(rng.integers(1, 6))
        if rng.random() < 0.5:
            cv2.line(img, pt1, pt2, color, thickness)
        else:
            cv2.circle(img, pt1, int(rng.integers(5, 30)), color, -1)
    return img


@pytest.mark.parametrize("true_angle", [0.0, 37.0, 145.0, 260.0])
def test_match_recovers_known_rotation_and_position(true_angle):
    target = _synthetic_textured_image()
    cx, cy, half = 450, 450, 55
    patch = target[cy - half:cy + half, cx - half:cx + half].copy()
    mask = np.full(patch.shape[:2], 255, np.uint8)
    # asymmetric notch so rotation isn't ambiguous
    mask[:20, :20] = 0
    mask[-25:, -15:] = 0

    rot_patch, rot_mask = _rotate_with_mask(patch, mask, true_angle)
    search_rect = (250, 250, 400, 400)

    result = match_piece_to_target(rot_patch, rot_mask, target, search_rect,
                                    scale_photo_per_target=1.0,
                                    coarse_step=15, fine_step=3, fine_range=16)
    assert result is not None
    angle, score, _combined, _cdist, (tx, ty), _size = result

    assert np.hypot(tx - cx, ty - cy) < 2.0
    expected_angle = (-true_angle) % 360.0
    angle_err = min((angle - expected_angle) % 360.0, (expected_angle - angle) % 360.0)
    assert angle_err <= 3.0
    assert score > 0.5


def test_valid_mask_prevents_matching_outside_picture_quad():
    """Regression test for a real bug found on an actual puzzle photo: search_rect
    is an axis-aligned bounding box around the (possibly tilted) picture quad, so
    its corners can fall outside the true picture - e.g. into a box's cardboard
    border. If something there happens to correlate strongly with a piece (as a
    blurry/glary border did for a real piece), the match lands in nonsense content
    instead of somewhere in the true picture. build_valid_mask must prevent that.

    Setup: the piece's real content is planted ONLY in a "leak" corner outside a
    tilted quad (inside the quad is unrelated noise), so an unconstrained search
    is guaranteed to land in the leak zone - then a masked search must not.
    """
    size = 500
    target = np.zeros((size, size, 3), np.uint8)
    rng = np.random.default_rng(7)
    target[:] = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)

    center = np.array([250.0, 250.0])
    half = 130
    square = np.array([[-half, -half], [half, -half], [half, half], [-half, half]],
                       dtype=np.float64)
    theta = np.radians(25)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    quad = (square @ R.T) + center

    # a piece cut from elsewhere entirely (never placed inside the quad at all)
    source = np.zeros((size, size, 3), np.uint8)
    for _ in range(150):
        pt1 = tuple(rng.integers(0, size, size=2).tolist())
        pt2 = tuple((np.array(pt1) + rng.integers(-40, 40, size=2)).tolist())
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.line(source, pt1, pt2, color, 2)
    half_p = 25
    piece = source[100 - half_p:100 + half_p, 100 - half_p:100 + half_p].copy()
    piece_mask = np.full(piece.shape[:2], 255, np.uint8)
    piece_mask[:8, :8] = 0  # asymmetric notch so rotation isn't ambiguous

    # plant it (unrotated - a trivial, unambiguous near-perfect match) in a
    # bounding-box corner that's outside the tilted quad
    bbox_x0, bbox_y0 = int(quad[:, 0].min()), int(quad[:, 1].min())
    leak_x, leak_y = bbox_x0 + 5, bbox_y0 + 5
    ph, pw = piece_mask.shape[:2]
    assert cv2.pointPolygonTest(
        quad.astype(np.float32), (leak_x + pw / 2, leak_y + ph / 2), False
    ) < 0, "test setup bug: leak zone should be outside the quad"
    region = target[leak_y:leak_y + ph, leak_x:leak_x + pw]
    region[piece_mask > 0] = piece[piece_mask > 0]

    search_rect = (bbox_x0, bbox_y0,
                   int(quad[:, 0].max()) - bbox_x0, int(quad[:, 1].max()) - bbox_y0)

    result_no_mask = match_piece_to_target(piece, piece_mask, target, search_rect,
                                            scale_photo_per_target=1.0,
                                            coarse_step=20, fine_step=4, fine_range=10)
    _angle, score_no_mask, _combined, _cdist, (mx, my), _size = result_no_mask
    assert cv2.pointPolygonTest(quad.astype(np.float32), (mx, my), False) < 0, (
        "test setup bug: an unconstrained search should find the only real match, "
        "which is in the leak zone")
    assert score_no_mask > 0.7

    valid_mask = build_valid_mask(target.shape, quad, inset_px=0)
    result_with_mask = match_piece_to_target(piece, piece_mask, target, search_rect,
                                              scale_photo_per_target=1.0,
                                              coarse_step=20, fine_step=4, fine_range=10,
                                              valid_mask=valid_mask)
    _angle2, score_with_mask, _combined2, _cdist2, (mx2, my2), _size2 = result_with_mask
    assert cv2.pointPolygonTest(quad.astype(np.float32), (mx2, my2), False) >= 0, (
        "masked search must not return a location outside the picture quad")
    assert score_with_mask < score_no_mask - 0.3, (
        "masked search should fall back to a much weaker match against unrelated "
        "noise inside the quad, not sneak out to the leak zone")


def test_color_correction_prefers_correct_hue_over_stronger_structural_match():
    """Regression test for two compounding real bugs found on an actual puzzle
    photo (arrows pointing at spots with an obviously wrong color, e.g. a green
    piece assigned to a patch of blue sky):

    1. matchTemplate's normalized cross-correlation is surprisingly tolerant of
       hue mismatches - a modest structural imperfection in the TRUE match
       (segmentation noise, JPEG artifacts) can easily lose to a pixel-perfect
       but wrong-colored decoy on raw NCC alone.
    2. The fine-search window around each coarse candidate was far larger than
       the minimum separation between distinct coarse candidates, so two
       genuinely different candidates' windows overlapped and both collapsed
       onto whichever one had the single strongest peak - silently defeating
       per-candidate color re-ranking even once it existed, because every
       "distinct" candidate re-found the exact same location.
    """
    rng = np.random.default_rng(3)
    size = 150
    base = np.zeros((size, size, 3), np.uint8)
    for _ in range(60):
        pt1 = tuple(rng.integers(0, size, size=2).tolist())
        pt2 = tuple((np.array(pt1) + rng.integers(-30, 30, size=2)).tolist())
        gray = int(rng.integers(80, 220))
        line_color = (gray * 0.3, gray * 0.9, gray * 0.3)  # greenish
        cv2.line(base, pt1, pt2, line_color, 2)

    half_p = 25
    pcx, pcy = 75, 75
    piece = base[pcy - half_p:pcy + half_p, pcx - half_p:pcx + half_p].copy()
    piece_mask = np.full(piece.shape[:2], 255, np.uint8)
    piece_mask[:6, :6] = 0
    ph, pw = piece.shape[:2]

    target = np.zeros((600, 600, 3), np.uint8)
    target[:] = rng.integers(60, 100, size=(600, 600, 3), dtype=np.uint8)

    # true location: correct color, but imperfect (blurred) - a stand-in for
    # ordinary segmentation/compression noise on a real photo
    true_cx, true_cy = 200, 200
    perturbed = cv2.GaussianBlur(piece, (0, 0), 2.0)
    region = target[true_cy - ph // 2:true_cy - ph // 2 + ph, true_cx - pw // 2:true_cx - pw // 2 + pw]
    region[piece_mask > 0] = perturbed[piece_mask > 0]

    # decoy: pixel-perfect structural copy, hue-shifted by 30/180 (a clearly
    # visible color mismatch, not lighting noise) - scores higher raw NCC
    # than the imperfect true match despite being the wrong color entirely
    hsv = cv2.cvtColor(piece, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[..., 0] = (hsv[..., 0] + 30) % 180
    decoy = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    decoy_cx, decoy_cy = 400, 400
    region2 = target[decoy_cy - ph // 2:decoy_cy - ph // 2 + ph, decoy_cx - pw // 2:decoy_cx - pw // 2 + pw]
    region2[piece_mask > 0] = decoy[piece_mask > 0]

    search_rect = (0, 0, 600, 600)

    result = match_piece_to_target(piece, piece_mask, target, search_rect,
                                    scale_photo_per_target=1.0,
                                    coarse_step=20, fine_step=4, fine_range=10)
    _angle, ncc, _combined, _cdist, (tx, ty), _size = result
    assert np.hypot(tx - decoy_cx, ty - decoy_cy) < 5.0, (
        "test setup bug: without color info, raw NCC should prefer the decoy")

    zero_correction = np.zeros(3, dtype=np.float32)
    result2 = match_piece_to_target(piece, piece_mask, target, search_rect,
                                     scale_photo_per_target=1.0,
                                     coarse_step=20, fine_step=4, fine_range=10,
                                     color_correction=zero_correction)
    _angle2, _ncc2, _combined2, color_dist2, (tx2, ty2), _size2 = result2
    assert np.hypot(tx2 - true_cx, ty2 - true_cy) < 5.0, (
        "color-aware search should prefer the correctly-colored (if slightly "
        "imperfect) true match over the wrong-colored decoy")
    assert color_dist2 < 5.0


def test_match_scores_true_match_higher_than_unrelated_piece():
    """An unrelated patch can still score moderately on a busy synthetic image by
    chance (some random line/circle alignment partially lines up) - the meaningful
    signal is that a true match scores clearly higher, which is what the
    render/confidence-tier step relies on."""
    target = _synthetic_textured_image(seed=1)
    other = _synthetic_textured_image(seed=2)
    unrelated_piece = other[400:460, 400:460].copy()
    mask = np.full(unrelated_piece.shape[:2], 255, np.uint8)

    search_rect = (0, 0, target.shape[1], target.shape[0])
    result = match_piece_to_target(unrelated_piece, mask, target, search_rect,
                                    scale_photo_per_target=1.0,
                                    coarse_step=20, fine_step=4, fine_range=10)
    assert result is not None
    _angle, unrelated_score, _combined, _cdist, _xy, _size = result

    true_piece = target[500:560, 500:560].copy()
    result2 = match_piece_to_target(true_piece, mask, target, search_rect,
                                     scale_photo_per_target=1.0,
                                     coarse_step=20, fine_step=4, fine_range=10)
    assert result2 is not None
    _angle2, true_score, _combined2, _cdist2, _xy2, _size2 = result2

    assert true_score > unrelated_score + 0.2


def test_restrict_to_gaps_ands_photo_space_mask_into_target_space():
    size = 200
    valid_mask_target = np.full((size, size), 255, np.uint8)  # everywhere valid to start

    gap_mask_photo = np.zeros((size, size), np.uint8)
    gap_mask_photo[50:100, 50:100] = 255  # only this region is a real gap

    identity_h = np.eye(3, dtype=np.float64)
    combined = restrict_to_gaps(valid_mask_target, gap_mask_photo, identity_h, (size, size))

    assert combined[75, 75] > 0, "inside the gap should stay valid"
    assert combined[10, 10] == 0, "outside the gap (already filled) must be excluded"
    assert combined[150, 150] == 0


class _IdentityAlignment:
    homography = np.eye(3, dtype=np.float64)


def test_match_all_confines_a_border_piece_to_the_border_band():
    """A piece with a detected straight edge is physically a border piece,
    full stop - regardless of what its printed content matches best. If its
    true content only appears (or best matches) in the picture's interior,
    that's a sign of a shape-classification error or a coincidental content
    match, not a valid destination, and the border constraint should keep
    match_all from reporting it anyway."""
    size = 400
    target = _synthetic_textured_image(seed=5, size=size)

    # plant an exact, otherwise-perfect copy of the piece's content at the
    # picture's dead center - deep inside the interior, far from any border
    half = 20
    cx, cy = size // 2, size // 2
    piece_bgr = target[cy - half:cy + half, cx - half:cx + half].copy()
    piece_mask_local = np.full(piece_bgr.shape[:2], 255, np.uint8)

    full_mask = np.zeros((size, size), np.uint8)
    full_mask[cy - half:cy + half, cx - half:cx + half] = piece_mask_local
    border_piece = Piece(id=1, mask=full_mask, bbox=(cx - half, cy - half, 2 * half, 2 * half),
                          centroid=(float(cx), float(cy)), angle_hint=0.0,
                          straight_edge_count=1)

    polygon = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float64)
    valid_mask = build_valid_mask((size, size), polygon, inset_px=0)
    # the valid-mask check requires the piece's WHOLE rotated bounding box to
    # fit inside the band (see _correlation_map), which at a 45-degree
    # rotation is the piece's diagonal (2*half*sqrt(2) = ~57px here) - the
    # band must be comfortably wider than that or no placement is ever legal
    # at any rotation and match_piece_to_target returns None for every angle
    band_px = 80
    border_mask = build_border_band_mask((size, size), polygon, band_px)

    # sanity: the planted content's own location must NOT be in the band
    assert border_mask[cy, cx] == 0

    matches = match_all([border_piece], target, target, _IdentityAlignment(),
                         search_rect=(0, 0, size, size), approx_target_point=(cx, cy),
                         valid_mask=valid_mask, border_mask=border_mask,
                         coarse_step=20, fine_step=4, fine_range=10)

    assert len(matches) == 1
    m = matches[0]
    tx, ty = m.target_xy
    assert border_mask[int(ty), int(tx)] > 0, (
        f"border piece matched outside the border band at ({tx:.0f},{ty:.0f})")
    assert m.ncc_score < 0.9, (
        "should NOT have found the planted perfect match at the center - "
        "the border constraint should have kept it out of reach")


def test_build_corner_regions_mask_covers_only_the_four_corners():
    size = 300
    polygon = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float64)
    mask = build_corner_regions_mask((size, size), polygon, radius_px=20)

    for cx, cy in [(0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)]:
        assert mask[cy, cx] > 0, f"corner ({cx},{cy}) should be covered"
    assert mask[size // 2, size // 2] == 0, "center must not be covered"
