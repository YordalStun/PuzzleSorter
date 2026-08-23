import numpy as np

from puzzlesorter.pieces import build_roi_mask, detect_pieces


def _stamp_textured_blob(img, cx, cy, size, rng):
    half = size // 2
    img[cy - half:cy + half, cx - half:cx + half] = rng.integers(
        0, 255, size=(2 * half, 2 * half, 3), dtype=np.uint8)


def test_detects_separated_pieces_and_ignores_plain_background():
    rng = np.random.default_rng(0)
    img = np.full((600, 800, 3), 180, np.uint8)
    piece_centers = [(100, 100), (300, 120), (500, 400), (700, 300), (200, 500)]
    piece_size = 40
    for cx, cy in piece_centers:
        _stamp_textured_blob(img, cx, cy, piece_size, rng)

    roi = build_roi_mask(img.shape)
    pieces = detect_pieces(img, roi, expected_piece_area=piece_size * piece_size)

    assert len(pieces) == len(piece_centers)
    found_centers = sorted(p.centroid for p in pieces)
    expected_centers = sorted(piece_centers)
    for (fx, fy), (ex, ey) in zip(found_centers, expected_centers):
        assert abs(fx - ex) < 6
        assert abs(fy - ey) < 6


def test_exclude_rects_and_margins_remove_regions_from_search():
    rng = np.random.default_rng(1)
    img = np.full((400, 400, 3), 180, np.uint8)
    _stamp_textured_blob(img, 50, 50, 30, rng)     # inside top margin -> excluded
    _stamp_textured_blob(img, 200, 200, 30, rng)   # kept
    _stamp_textured_blob(img, 350, 350, 30, rng)   # inside exclude rect -> excluded

    roi = build_roi_mask(img.shape, exclude_rects=[(300, 300, 100, 100)], margin_top=100)
    pieces = detect_pieces(img, roi, expected_piece_area=30 * 30)

    assert len(pieces) == 1
    assert abs(pieces[0].centroid[0] - 200) < 6
    assert abs(pieces[0].centroid[1] - 200) < 6
