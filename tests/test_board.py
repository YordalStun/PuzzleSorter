import numpy as np
import cv2

from puzzlesorter.board import find_assembled_region, find_gaps


def test_finds_textured_rectangle_on_plain_background():
    img = np.full((600, 800, 3), 180, np.uint8)  # plain gray background
    rng = np.random.default_rng(0)
    x0, y0, w, h = 200, 150, 350, 250
    img[y0:y0 + h, x0:x0 + w] = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)

    region = find_assembled_region(img)
    x, y, rw, rh = cv2.boundingRect(region.contour)

    assert abs(x - x0) < 15
    assert abs(y - y0) < 15
    assert abs((x + rw) - (x0 + w)) < 15
    assert abs((y + rh) - (y0 + h)) < 15


def test_find_gaps_locates_unfilled_patch_within_assembled_region():
    """Regression test for a real bug found on an actual puzzle photo: a match
    was pointed at a spot that already had a real piece placed there, because
    nothing checked whether a specific spot *within* the assembled region was
    actually still empty - find_assembled_region only sees the region's outer
    extent (RETR_EXTERNAL doesn't record internal holes)."""
    img = np.full((600, 800, 3), 180, np.uint8)
    rng = np.random.default_rng(1)
    bx0, by0, bw, bh = 100, 100, 500, 350
    img[by0:by0 + bh, bx0:bx0 + bw] = rng.integers(30, 255, size=(bh, bw, 3), dtype=np.uint8)

    # carve an unfilled gap (back to plain background color) inside the block
    gx0, gy0, gw, gh = 300, 200, 120, 90
    img[gy0:gy0 + gh, gx0:gx0 + gw] = 180

    board_quad = np.array([[bx0, by0], [bx0 + bw, by0], [bx0 + bw, by0 + bh], [bx0, by0 + bh]],
                           dtype=np.float32)
    gaps = find_gaps(img, board_quad, erode_edge_px=5)

    gap_center_val = gaps[gy0 + gh // 2, gx0 + gw // 2]
    filled_center_val = gaps[by0 + 30, bx0 + 30]
    assert gap_center_val > 0, "the carved-out gap should be detected as a gap"
    assert filled_center_val == 0, "the filled interior should not be flagged as a gap"
