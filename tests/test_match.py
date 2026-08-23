import numpy as np
import cv2
import pytest

from puzzlesorter.match import match_piece_to_target, _rotate_with_mask


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
