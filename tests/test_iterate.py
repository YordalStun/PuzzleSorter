import numpy as np
import cv2

from puzzlesorter.iterate import run_iterative_solve
from puzzlesorter.pieces import Piece


def _synthetic_textured_image(size=400, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    for _ in range(300):
        pt1 = tuple(rng.integers(0, size, size=2).tolist())
        pt2 = tuple((np.array(pt1) + rng.integers(-40, 40, size=2)).tolist())
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.line(img, pt1, pt2, color, 2)
    return img


class _IdentityAlignment:
    """homography=identity so target space == photo space, which makes it
    straightforward to construct pieces whose true location is known."""
    homography = np.eye(3, dtype=np.float64)


def _make_piece(next_id, photo, cx, cy, half=20):
    crop = photo[cy - half:cy + half, cx - half:cx + half].copy()
    mask = np.full(crop.shape[:2], 255, np.uint8)
    full_mask = np.zeros(photo.shape[:2], np.uint8)
    full_mask[cy - half:cy + half, cx - half:cx + half] = mask
    return Piece(id=next_id, mask=full_mask, bbox=(cx - half, cy - half, 2 * half, 2 * half),
                 centroid=(float(cx), float(cy)), angle_hint=0.0)


def test_places_exact_matches_and_shrinks_remaining_pool():
    target = _synthetic_textured_image()
    photo = target.copy()  # identity alignment: photo IS the target here

    # two pieces, each an exact crop of a known target region far from each other
    piece_a = _make_piece(1, photo, 100, 100)
    piece_b = _make_piece(2, photo, 300, 300)

    valid_mask = np.full(target.shape[:2], 255, np.uint8)  # everywhere is a "gap"
    avg_piece_area_target = (40 * 40)

    rounds = run_iterative_solve(
        [piece_a, piece_b], photo, target, _IdentityAlignment(),
        search_rect=(0, 0, target.shape[1], target.shape[0]),
        valid_mask=valid_mask, approx_target_point=(200, 200),
        avg_piece_area_target=avg_piece_area_target,
        auto_place_threshold=0.5, max_rounds=3,
        match_kwargs=dict(coarse_step=20, fine_step=4, fine_range=10),
    )

    assert len(rounds) >= 1
    placed_ids = {m.piece_id for r in rounds for m in r.placed}
    assert placed_ids == {1, 2}, "both exact-copy pieces should clear the auto-place bar"

    # the loop should stop once nothing remains, not burn through max_rounds
    assert rounds[-1].placed or len(rounds) == 1
    total_considered_first_round = len(rounds[0].considered_piece_ids)
    assert total_considered_first_round == 2

    # a piece placed in an earlier round must never be offered again later
    placed_so_far = set()
    for r in rounds:
        assert not (placed_so_far & set(r.considered_piece_ids)), (
            f"round {r.number} reconsidered an already-placed piece")
        placed_so_far |= {m.piece_id for m in r.placed}


def test_stops_when_nothing_clears_the_threshold():
    target = _synthetic_textured_image(seed=2)
    photo = target.copy()

    # a piece that does NOT actually appear anywhere in the target/photo
    other = _synthetic_textured_image(seed=3)
    half = 20
    crop = other[100 - half:100 + half, 100 - half:100 + half]
    full_mask = np.zeros(photo.shape[:2], np.uint8)
    full_mask[100 - half:100 + half, 100 - half:100 + half] = 255
    unrelated_piece = Piece(id=1, mask=full_mask, bbox=(80, 80, 40, 40),
                             centroid=(100.0, 100.0), angle_hint=0.0)
    # graft the unrelated crop's pixels into the photo at its own bbox so
    # match_all reads real (but non-matching) pixel content there
    photo = photo.copy()
    photo[100 - half:100 + half, 100 - half:100 + half] = crop

    valid_mask = np.full(target.shape[:2], 255, np.uint8)
    rounds = run_iterative_solve(
        [unrelated_piece], photo, target, _IdentityAlignment(),
        search_rect=(0, 0, target.shape[1], target.shape[0]),
        valid_mask=valid_mask, approx_target_point=(200, 200),
        avg_piece_area_target=(40 * 40),
        auto_place_threshold=0.9, max_rounds=3,
        match_kwargs=dict(coarse_step=20, fine_step=4, fine_range=10),
    )

    assert len(rounds) == 1, "should stop after the first round places nothing"
    assert rounds[0].placed == []
