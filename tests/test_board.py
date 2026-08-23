import numpy as np
import cv2

from puzzlesorter.board import find_assembled_region


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
