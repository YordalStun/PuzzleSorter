"""PuzzleSorter: figure out where loose jigsaw puzzle pieces go, from photos."""
from .align import Alignment, align_target_to_photo, warp_target_into_photo
from .board import BoardRegion, find_assembled_region, find_gaps
from .color import estimate_color_correction
from .iterate import Round, run_iterative_solve
from .pieces import Piece, build_roi_mask, detect_pieces
from .shape import PieceShape, EdgeShape, classify_piece_shape
from .match import (Match, match_all, flag_conflicts, build_valid_mask, restrict_to_gaps,
                     build_border_band_mask, build_corner_regions_mask)
from .visualize import render_solution, render_batches, write_csv

__all__ = [
    "Alignment", "align_target_to_photo", "warp_target_into_photo",
    "BoardRegion", "find_assembled_region", "find_gaps",
    "estimate_color_correction",
    "Round", "run_iterative_solve",
    "Piece", "build_roi_mask", "detect_pieces",
    "PieceShape", "EdgeShape", "classify_piece_shape",
    "Match", "match_all", "flag_conflicts", "build_valid_mask", "restrict_to_gaps",
    "build_border_band_mask", "build_corner_regions_mask",
    "render_solution", "render_batches", "write_csv",
]
