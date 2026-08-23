import numpy as np
import cv2
import pytest

from puzzlesorter.match import match_piece_to_target, _rotate_with_mask, build_valid_mask


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
    angle, score, (tx, ty), _size = result

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
    _angle, score_no_mask, (mx, my), _size = result_no_mask
    assert cv2.pointPolygonTest(quad.astype(np.float32), (mx, my), False) < 0, (
        "test setup bug: an unconstrained search should find the only real match, "
        "which is in the leak zone")
    assert score_no_mask > 0.7

    valid_mask = build_valid_mask(target.shape, quad, inset_px=0)
    result_with_mask = match_piece_to_target(piece, piece_mask, target, search_rect,
                                              scale_photo_per_target=1.0,
                                              coarse_step=20, fine_step=4, fine_range=10,
                                              valid_mask=valid_mask)
    _angle2, score_with_mask, (mx2, my2), _size2 = result_with_mask
    assert cv2.pointPolygonTest(quad.astype(np.float32), (mx2, my2), False) >= 0, (
        "masked search must not return a location outside the picture quad")
    assert score_with_mask < score_no_mask - 0.3, (
        "masked search should fall back to a much weaker match against unrelated "
        "noise inside the quad, not sneak out to the leak zone")


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
    _angle, unrelated_score, _xy, _size = result

    true_piece = target[500:560, 500:560].copy()
    result2 = match_piece_to_target(true_piece, mask, target, search_rect,
                                     scale_photo_per_target=1.0,
                                     coarse_step=20, fine_step=4, fine_range=10)
    assert result2 is not None
    _angle2, true_score, _xy2, _size2 = result2

    assert true_score > unrelated_score + 0.2
