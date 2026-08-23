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

from .align import Alignment, warp_target_into_photo
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


def _piece_silhouette_photo(piece, match, photo_shape):
    """The piece's own segmented silhouette (piece.mask, already full-photo
    sized), rotated to its correct board orientation and translated to its
    destination - the real rotated shape (tabs, blanks and all), not an
    ellipse approximation, so compositing and gap bookkeeping both reflect
    the piece's actual footprint."""
    cx0, cy0 = piece.centroid
    M = cv2.getRotationMatrix2D((cx0, cy0), match.rotation_degrees, 1.0)
    M[0, 2] += match.photo_xy[0] - cx0
    M[1, 2] += match.photo_xy[1] - cy0
    h, w = photo_shape[:2]
    return cv2.warpAffine(piece.mask, M, (w, h), flags=cv2.INTER_NEAREST)


def _composite_placements(composite_photo, warped_target, piece_by_id, placed):
    out = composite_photo.copy()
    for m in placed:
        footprint = _piece_silhouette_photo(piece_by_id[m.piece_id], m, out.shape[:2])
        out[footprint > 0] = warped_target[footprint > 0]
    return out


def _subtract_placed_from_gaps(gap_mask_target, piece_by_id, placed, homography, photo_shape):
    out = gap_mask_target.copy()
    Hinv = np.linalg.inv(homography)
    th, tw = out.shape[:2]
    for m in placed:
        footprint_photo = _piece_silhouette_photo(piece_by_id[m.piece_id], m, photo_shape)
        # dilate slightly before projecting into target space: gap bookkeeping
        # erring a little generous is preferable to leaving an unmatchable
        # sliver right at the piece's true edge from segmentation imprecision
        footprint_photo = cv2.dilate(footprint_photo, np.ones((5, 5), np.uint8))
        footprint_target = cv2.warpPerspective(footprint_photo, Hinv, (tw, th),
                                                flags=cv2.INTER_NEAREST)
        out[footprint_target > 0] = 0
    return out


def run_iterative_solve(pieces: List[Piece], photo_bgr, target_bgr, alignment: Alignment,
                         search_rect, valid_mask, approx_target_point, avg_piece_area_target,
                         color_correction=None, auto_place_threshold=0.55, max_rounds=6,
                         border_mask=None, corner_mask=None,
                         match_kwargs=None, on_round_complete=None) -> List[Round]:
    """Run up to max_rounds of: match everything still loose, virtually commit
    the matches that clear auto_place_threshold and aren't conflicted, update
    the working photo and remaining gap mask, repeat for whatever's left.
    Stops early if nothing remains, or a round places nothing new.

    on_round_complete, if given, is called with each Round as it finishes -
    e.g. a print callback for progress on a run long enough to want it.

    border_mask/corner_mask (see match.build_border_band_mask/build_corner_regions_mask):
    forwarded unchanged into every round's match_all call, so a piece with a
    detected straight edge stays confined to the border/corner region across
    every round, not just the first.
    """
    match_kwargs = dict(match_kwargs or {})
    piece_by_id = {p.id: p for p in pieces}
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
                             border_mask=border_mask, corner_mask=corner_mask,
                             color_correction=color_correction, **match_kwargs)
        conflicts = flag_conflicts(matches, min_distance=min_distance)
        placed = [m for m in matches
                  if m.score >= auto_place_threshold and not conflicts.get(m.piece_id)]

        composite = _composite_placements(composite, warped_target, piece_by_id, placed)
        gap_mask = _subtract_placed_from_gaps(gap_mask, piece_by_id, placed,
                                               alignment.homography, photo_bgr.shape)

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
