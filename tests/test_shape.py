import numpy as np
import cv2
import pytest

from puzzlesorter.shape import classify_piece_shape

EXPECTED_PIECE_SIZE = 140


def _make_piece(size=200, top='straight', right='tab', bottom='blank', left='tab', bump_r=22):
    mask = np.zeros((size, size), np.uint8)
    pad = (size - EXPECTED_PIECE_SIZE) // 2
    cv2.rectangle(mask, (pad, pad), (size - pad, size - pad), 255, -1)
    mid = size // 2
    specs = {'top': (mid, pad, top), 'bottom': (mid, size - pad, bottom),
             'left': (pad, mid, left), 'right': (size - pad, mid, right)}
    for _name, (cx, cy, kind) in specs.items():
        if kind == 'straight':
            continue
        elif kind == 'tab':
            cv2.circle(mask, (cx, cy), bump_r, 255, -1)
        elif kind == 'blank':
            cv2.circle(mask, (cx, cy), bump_r, 0, -1)
    return mask


# box edge order for an axis-aligned piece comes out as (bottom, left, top, right)
BOX_ORDER = ['bottom', 'left', 'top', 'right']


@pytest.mark.parametrize("cfg,expected_straight,expected_border,expected_corner", [
    (dict(top='straight', right='tab', bottom='blank', left='tab'), 1, True, False),
    (dict(top='straight', right='straight', bottom='tab', left='blank'), 2, False, True),
    (dict(top='tab', right='blank', bottom='tab', left='blank'), 0, False, False),
    (dict(top='straight', right='blank', bottom='straight', left='tab'), 2, False, True),
    (dict(top='tab', right='tab', bottom='tab', left='tab'), 0, False, False),
])
def test_classifies_edge_kinds_and_derived_flags(cfg, expected_straight, expected_border,
                                                   expected_corner):
    mask = _make_piece(**cfg)
    shape = classify_piece_shape(mask, expected_piece_size=EXPECTED_PIECE_SIZE)

    got_kinds = [e.kind for e in shape.edges]
    expected_kinds = [cfg[side] for side in BOX_ORDER]
    assert got_kinds == expected_kinds

    assert shape.straight_count == expected_straight
    assert shape.is_border == expected_border
    assert shape.is_corner == expected_corner


def test_tab_and_blank_deviations_roughly_match_true_bump_radius():
    bump_r = 22
    mask = _make_piece(top='tab', right='blank', bottom='tab', left='blank', bump_r=bump_r)
    shape = classify_piece_shape(mask, expected_piece_size=EXPECTED_PIECE_SIZE)
    for e in shape.edges:
        assert e.kind in ("tab", "blank")
        assert abs(e.deviation - bump_r) < 8, f"deviation {e.deviation} far from true {bump_r}"


def test_robust_to_small_boundary_jitter():
    """Real segmented pieces have a few pixels of jitter from JPEG noise and
    thresholding - that alone shouldn't read as a tab/blank on a straight
    edge."""
    rng = np.random.default_rng(0)
    mask = _make_piece(top='straight', right='straight', bottom='straight', left='straight')

    # confine noise to a thin band straddling the actual boundary, unlike a
    # stray circle dropped anywhere in the canvas (including deep inside the
    # piece), which wouldn't model boundary jitter at all
    band = cv2.subtract(cv2.dilate(mask, np.ones((7, 7), np.uint8)),
                         cv2.erode(mask, np.ones((7, 7), np.uint8)))
    band_ys, band_xs = np.nonzero(band)

    noisy = mask.copy()
    for _ in range(60):
        idx = rng.integers(0, len(band_xs))
        cx, cy = int(band_xs[idx]), int(band_ys[idx])
        r = int(rng.integers(1, 3))
        color = 255 if rng.random() < 0.5 else 0
        cv2.circle(noisy, (cx, cy), r, color, -1)

    shape = classify_piece_shape(noisy, expected_piece_size=EXPECTED_PIECE_SIZE)
    assert shape.straight_count == 4, (
        f"boundary jitter should not be mistaken for tabs/blanks: {[e.kind for e in shape.edges]}")
