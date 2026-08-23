"""PuzzleSorter: figure out where loose jigsaw puzzle pieces go, from photos."""
from .align import Alignment, align_target_to_photo, warp_target_into_photo
from .board import BoardRegion, find_assembled_region, find_gaps
from .color import estimate_color_correction
from .pieces import Piece, build_roi_mask, detect_pieces
from .match import Match, match_all, flag_conflicts, build_valid_mask, restrict_to_gaps
from .visualize import render_solution, render_batches, write_csv

__all__ = [
    "Alignment", "align_target_to_photo", "warp_target_into_photo",
    "BoardRegion", "find_assembled_region", "find_gaps",
    "estimate_color_correction",
    "Piece", "build_roi_mask", "detect_pieces",
    "Match", "match_all", "flag_conflicts", "build_valid_mask", "restrict_to_gaps",
    "render_solution", "render_batches", "write_csv",
]
