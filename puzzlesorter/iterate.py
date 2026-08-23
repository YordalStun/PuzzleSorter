"""Simulate placing high-confidence pieces and re-solving for what's left,
without needing a new photo from the user after each round.

This does not know whether the pieces were actually placed for real - it
assumes every auto-placed match from a round is correct and commits to that
before computing the next round. Wrong assumptions compound, which is why
placement is restricted to a confidence bar at least as strict as what's
shown to the user as "high confidence", and each round is reported
separately so a human can sanity-check before trusting the next one.
"""
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .align import Alignment, local_affine, warp_target_into_photo
from .match import Match, match_all, flag_conflicts
from .pieces import Piece


@dataclass
class Round:
    number: int
    considered_piece_ids: List[int]
    matches: List[Match]
    conflicts: dict
    placed: List[Match]  # the subset of matches auto-committed this round
    composite_photo: np.ndarray  # working photo AFTER this round's placements


def _footprint_ellipse(mask_shape, center_xy, size_wh, scale=1.0):
    """A filled-ellipse mask approximating a piece's footprint - good enough
    for gap bookkeeping and visualization; doesn't need to be the exact
    rotated silhouette, since slight over/under-marking just leaves a small
    sliver for the next round to sort out either way."""
    mask = np.zeros(mask_shape, np.uint8)
    axes = (max(1, int(size_wh[0] * scale / 2)), max(1, int(size_wh[1] * scale / 2)))
    cv2.ellipse(mask, (int(center_xy[0]), int(center_xy[1])), axes, 0, 0, 360, 255, -1)
    return mask


def _composite_placements(composite_photo, warped_target, alignment, placed):
    out = composite_photo.copy()
    for m in placed:
        scale, _ = local_affine(alignment, m.target_xy)
        photo_size = (m.target_size[0] * scale, m.target_size[1] * scale)
        footprint = _footprint_ellipse(out.shape[:2], m.photo_xy, photo_size, scale=1.0)
        out[footprint > 0] = warped_target[footprint > 0]
    return out


def _subtract_placed_from_gaps(gap_mask_target, placed):
    out = gap_mask_target.copy()
    for m in placed:
        footprint = _footprint_ellipse(out.shape[:2], m.target_xy, m.target_size, scale=1.15)
        out[footprint > 0] = 0
    return out


def run_iterative_solve(pieces: List[Piece], photo_bgr, target_bgr, alignment: Alignment,
                         search_rect, valid_mask, approx_target_point, avg_piece_area_target,
                         color_correction=None, auto_place_threshold=0.40, max_rounds=6,
                         match_kwargs=None, on_round_complete=None) -> List[Round]:
    """Run up to max_rounds of: match everything still loose, virtually commit
    the matches that clear auto_place_threshold and aren't conflicted, update
    the working photo and remaining gap mask, repeat for whatever's left.
    Stops early if nothing remains, or a round places nothing new.

    on_round_complete, if given, is called with each Round as it finishes -
    e.g. a print callback for progress on a run long enough to want it.
    """
    match_kwargs = dict(match_kwargs or {})
    remaining = {p.id: p for p in pieces}
    gap_mask = valid_mask.copy()
    composite = photo_bgr.copy()
    warped_target = warp_target_into_photo(target_bgr, photo_bgr, alignment)
    min_distance = 0.5 * np.sqrt(avg_piece_area_target)

    rounds = []
    for round_num in range(1, max_rounds + 1):
        if not remaining:
            break
        piece_list = list(remaining.values())
        matches = match_all(piece_list, composite, target_bgr, alignment, search_rect,
                             approx_target_point=approx_target_point, valid_mask=gap_mask,
                             color_correction=color_correction, **match_kwargs)
        conflicts = flag_conflicts(matches, min_distance=min_distance)
        placed = [m for m in matches
                  if m.score >= auto_place_threshold and not conflicts.get(m.piece_id)]

        composite = _composite_placements(composite, warped_target, alignment, placed)
        gap_mask = _subtract_placed_from_gaps(gap_mask, placed)

        this_round = Round(number=round_num, considered_piece_ids=list(remaining.keys()),
                            matches=matches, conflicts=conflicts, placed=placed,
                            composite_photo=composite)
        rounds.append(this_round)
        if on_round_complete:
            on_round_complete(this_round)

        for m in placed:
            remaining.pop(m.piece_id, None)

        if not placed:
            break

    return rounds
