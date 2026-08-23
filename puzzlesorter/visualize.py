"""Render the solved-piece suggestions as arrows/labels on the working photo."""
from typing import List

import cv2
import numpy as np

from .match import Match
from .pieces import Piece


def _confidence_tier(score, high=0.55, medium=0.35):
    if score >= high:
        return "high", (60, 200, 60)      # green (BGR)
    if score >= medium:
        return "medium", (0, 165, 255)    # orange
    return "low", (60, 60, 220)           # red


def render_solution(photo_bgr, pieces: List[Piece], matches: List[Match],
                     max_arrows=25, label_all=True, conflicts: dict = None):
    conflicts = conflicts or {}
    piece_by_id = {p.id: p for p in pieces}
    vis = photo_bgr.copy()
    overlay = photo_bgr.copy()

    ranked = sorted(matches, key=lambda m: -m.score)
    arrow_ids = {m.piece_id for m in ranked[:max_arrows]}

    for m in matches:
        piece = piece_by_id[m.piece_id]
        is_conflicted = conflicts.get(m.piece_id, False)
        tier, color = _confidence_tier(-1.0 if is_conflicted else m.score)
        px, py = piece.centroid
        dx, dy = m.photo_xy

        if label_all:
            cv2.circle(overlay, (int(px), int(py)), 10, color, 2)
            cv2.circle(overlay, (int(dx), int(dy)), 6, color, -1)
            cv2.putText(overlay, str(m.piece_id), (int(px) + 8, int(py) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            cv2.putText(overlay, str(m.piece_id), (int(dx) + 6, int(dy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            if is_conflicted:
                cv2.putText(overlay, "?", (int(px) - 16, int(py) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        if m.piece_id in arrow_ids and not is_conflicted:
            cv2.arrowedLine(overlay, (int(px), int(py)), (int(dx), int(dy)),
                             color, 2, cv2.LINE_AA, tipLength=0.05)

    vis = cv2.addWeighted(photo_bgr, 0.35, overlay, 0.65, 0)
    return vis


def write_csv(path, pieces: List[Piece], matches: List[Match], conflicts: dict = None):
    conflicts = conflicts or {}
    piece_by_id = {p.id: p for p in pieces}
    with open(path, "w") as f:
        f.write("piece_id,photo_x,photo_y,dest_photo_x,dest_photo_y,"
                "target_x,target_y,rotation_degrees,score,confidence,conflicted\n")
        for m in sorted(matches, key=lambda m: -m.score):
            piece = piece_by_id[m.piece_id]
            is_conflicted = conflicts.get(m.piece_id, False)
            tier, _ = _confidence_tier(-1.0 if is_conflicted else m.score)
            f.write(f"{m.piece_id},{piece.centroid[0]:.1f},{piece.centroid[1]:.1f},"
                    f"{m.photo_xy[0]:.1f},{m.photo_xy[1]:.1f},"
                    f"{m.target_xy[0]:.1f},{m.target_xy[1]:.1f},"
                    f"{m.rotation_degrees:.1f},{m.score:.4f},{tier},{is_conflicted}\n")
