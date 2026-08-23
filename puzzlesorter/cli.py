"""Command-line entry point: figure out where each loose jigsaw piece goes.

Example:
    python -m puzzlesorter.cli --unsolved photo.jpg --target box.jpg \\
        --pieces 500 --out result.jpg --csv result.csv
"""
import argparse
import os
import sys

import cv2
import numpy as np

from .align import align_target_to_photo, local_affine, warp_target_into_photo
from .board import find_assembled_region, find_gaps
from .color import estimate_color_correction
from .iterate import run_iterative_solve
from .pieces import build_roi_mask, detect_pieces
from .match import (match_all, flag_conflicts, build_valid_mask, restrict_to_gaps,
                     build_border_band_mask, build_corner_regions_mask)
from .visualize import render_solution, render_batches, write_csv


def _parse_rect(s):
    parts = [int(v) for v in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x,y,w,h")
    return tuple(parts)


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unsolved", required=True,
                   help="Photo of the puzzle in progress: assembled portion + loose pieces")
    p.add_argument("--target", required=True,
                   help="Reference image of the finished puzzle (e.g. a photo of the box)")
    p.add_argument("--out", default="solution.jpg", help="Output annotated image path")
    p.add_argument("--csv", default=None, help="Optional CSV path listing every piece's match")
    p.add_argument("--pieces", type=int, default=1000,
                   help="Total piece count for the puzzle (printed on the box). "
                        "Used only to estimate a single piece's pixel size; doesn't need "
                        "to be exact. Default: 1000")
    p.add_argument("--exclude", action="append", default=[], type=_parse_rect,
                   metavar="x,y,w,h",
                   help="Rectangle (in the --unsolved photo) to exclude from piece search, "
                        "e.g. a tool or box lying on the table. Repeatable.")
    p.add_argument("--margin-top", type=int, default=0)
    p.add_argument("--margin-bottom", type=int, default=0)
    p.add_argument("--margin-left", type=int, default=0)
    p.add_argument("--margin-right", type=int, default=0)
    p.add_argument("--max-arrows", type=int, default=25,
                   help="Only draw arrows for the N highest-confidence matches on the "
                        "single --out overview image; every match still gets a numbered "
                        "label there. Use --batch-size for clearer, less cluttered images.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Also write a series of clearer images with only this many "
                        "arrows each (highest-confidence batches first), named "
                        "<out>_batch01.jpg, <out>_batch02.jpg, ... Conflicted matches "
                        "are left out of batches. Recommended: 8-12.")
    p.add_argument("--min-inliers", type=int, default=15,
                   help="Minimum ORB inlier matches required to trust the target<->photo "
                        "alignment.")
    p.add_argument("--iterate", type=int, default=None, metavar="ROUNDS",
                   help="Simulate placing high-confidence pieces and re-solving for what's "
                        "left, for up to ROUNDS rounds, without needing a new photo between "
                        "rounds. Writes <out>_round01.jpg, <out>_round02.jpg, ... (each "
                        "round's newly-placed pieces only) plus a cumulative CSV. This "
                        "commits to each round's placements before computing the next, so "
                        "it only auto-places matches clearing --auto-place-threshold.")
    p.add_argument("--auto-place-threshold", type=float, default=0.45,
                   help="Minimum combined score for --iterate to treat a match as placed "
                        "rather than just suggested. Default 0.45 (stricter than the 0.40 "
                        "'high confidence' display cutoff, since a wrong auto-placement "
                        "affects every later round).")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    target = cv2.imread(args.target)
    if target is None:
        print(f"error: could not read target image: {args.target}", file=sys.stderr)
        return 1
    photo = cv2.imread(args.unsolved)
    if photo is None:
        print(f"error: could not read unsolved-puzzle image: {args.unsolved}", file=sys.stderr)
        return 1

    print("Aligning target image to the photo...")
    try:
        aln = align_target_to_photo(target, photo, min_inliers=args.min_inliers)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"  {aln.inlier_matches}/{aln.total_matches} feature matches agreed on alignment")

    print("Locating the already-assembled region...")
    try:
        board = find_assembled_region(photo)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    Hinv = np.linalg.inv(aln.homography)
    picture_quad_target = cv2.perspectiveTransform(
        board.quad.reshape(-1, 1, 2).astype(np.float32), Hinv
    ).reshape(-1, 2)
    pic_w, pic_h = cv2.minAreaRect(picture_quad_target.astype(np.float32))[1]
    avg_piece_area_target = (pic_w * pic_h) / max(1, args.pieces)
    board_center_target = tuple(picture_quad_target.mean(axis=0))
    scale = local_affine(aln, board_center_target)[0]
    expected_piece_area_photo = avg_piece_area_target * scale ** 2

    # a piece with a detected straight edge (see shape.classify_piece_shape) is
    # physically a border piece, and one with 2+ a corner piece - true regardless
    # of what its printed content matches best. The band/corner regions need to
    # be generously wider than one piece: match._correlation_map requires a
    # candidate rotation's WHOLE bounding box (up to the piece's diagonal, at a
    # 45-degree rotation) to fit inside, or that rotation has no valid position
    # at all for that piece. sqrt(avg_piece_area) is the diagonal of an
    # idealized SQUARE piece at the AVERAGE size - real pieces run bigger
    # (irregular tab/blank shapes, plus real size variance around the
    # average), so this needs real headroom above 1x; verified against an
    # actual photo that 1.6x still silently dropped real border pieces
    # whose true bounding box didn't fit, 2.2x recovered them. match_all
    # also falls back to the unconstrained search if the constrained one
    # finds nothing, so an unusually large piece still gets a match rather
    # than vanishing from the output.
    piece_span = float(np.sqrt(avg_piece_area_target))
    border_mask = build_border_band_mask(target.shape, picture_quad_target,
                                          band_px=int(round(2.2 * piece_span)))
    corner_mask = build_corner_regions_mask(target.shape, picture_quad_target,
                                             radius_px=int(round(2.2 * piece_span)))

    print("Detecting loose pieces...")
    roi = build_roi_mask(photo.shape, board_mask=board.mask, exclude_rects=args.exclude,
                          margin_top=args.margin_top, margin_bottom=args.margin_bottom,
                          margin_left=args.margin_left, margin_right=args.margin_right)
    pieces = detect_pieces(photo, roi, expected_piece_area_photo)
    n_corner = sum(1 for p in pieces if p.straight_edge_count >= 2)
    n_border = sum(1 for p in pieces if p.straight_edge_count == 1)
    print(f"  found {len(pieces)} candidate pieces "
          f"({n_corner} corner, {n_border} other border, "
          f"{len(pieces) - n_corner - n_border} interior, by detected edge shape)")
    if not pieces:
        print("error: no loose pieces detected - try adjusting --exclude/--margin-* "
              "to point the search at the right area of the photo", file=sys.stderr)
        return 1

    xs, ys = picture_quad_target[:, 0], picture_quad_target[:, 1]
    margin = 40
    sx, sy = max(0, int(xs.min() - margin)), max(0, int(ys.min() - margin))
    sx1 = min(target.shape[1], int(xs.max() + margin))
    sy1 = min(target.shape[0], int(ys.max() + margin))
    search_rect = (sx, sy, sx1 - sx, sy1 - sy)
    # search_rect is an axis-aligned box around a possibly-tilted picture quad, so its
    # corners can fall outside the true picture (the box's cardboard border, glare, a
    # hand holding it, etc.) - constrain matches to the quad itself to avoid landing there.
    valid_mask = build_valid_mask(target.shape, picture_quad_target)

    print("Finding already-filled spots within the assembled area...")
    gaps = find_gaps(photo, board.quad)
    valid_mask = restrict_to_gaps(valid_mask, gaps, aln.homography, target.shape)
    if not np.any(valid_mask):
        print("error: no empty gaps found within the assembled area - is the puzzle "
              "already complete, or did board detection go wrong?", file=sys.stderr)
        return 1

    print("Calibrating for lighting/white-balance differences between the two photos...")
    warped_target = warp_target_into_photo(target, photo, aln)
    filled_mask = cv2.bitwise_and(board.mask, cv2.bitwise_not(gaps))
    if np.count_nonzero(filled_mask) < 1000:
        print("  not enough already-assembled area to calibrate color; proceeding without it")
        color_correction = None
    else:
        color_correction = estimate_color_correction(photo, warped_target, filled_mask)

    if args.iterate:
        print(f"Simulating up to {args.iterate} rounds "
              f"(auto-place threshold {args.auto_place_threshold})...")

        def _report_round(r):
            print(f"  round {r.number}: considered {len(r.considered_piece_ids)}, "
                  f"placed {len(r.placed)}")

        rounds = run_iterative_solve(
            pieces, photo, target, aln, search_rect, valid_mask, board_center_target,
            avg_piece_area_target, color_correction=color_correction,
            auto_place_threshold=args.auto_place_threshold, max_rounds=args.iterate,
            border_mask=border_mask, corner_mask=corner_mask,
            on_round_complete=_report_round)

        stem, ext = os.path.splitext(args.out)
        all_placed = []
        for r in rounds:
            round_path = f"{stem}_round{r.number:02d}{ext}"
            vis = render_solution(r.composite_photo, pieces, r.placed, max_arrows=len(r.placed))
            cv2.imwrite(round_path, vis)
            all_placed.extend(r.placed)

        if not all_placed:
            print("error: no round placed anything above --auto-place-threshold - "
                  "try lowering it, or use single-round mode to see raw suggestions",
                  file=sys.stderr)
            return 1

        final_path = f"{stem}_final{ext}"
        cv2.imwrite(final_path, rounds[-1].composite_photo)
        print(f"Writing {final_path} (cumulative working photo after all rounds)")

        if args.csv:
            write_csv(args.csv, pieces, all_placed)
            print(f"Writing {args.csv} ({len(all_placed)} placed pieces across "
                  f"{len(rounds)} rounds)")

        still_loose = len(pieces) - len(all_placed)
        print(f"Placed {len(all_placed)} of {len(pieces)} candidate pieces "
              f"over {len(rounds)} round(s); {still_loose} still need a fresh photo "
              f"or manual placement.")
        print("Done.")
        return 0

    print(f"Matching {len(pieces)} pieces against the target image "
          f"(this can take a while)...")
    matches = match_all(pieces, photo, target, aln, search_rect,
                         approx_target_point=board_center_target, valid_mask=valid_mask,
                         border_mask=border_mask, corner_mask=corner_mask,
                         color_correction=color_correction)
    print(f"  matched {len(matches)} pieces")

    conflicts = flag_conflicts(matches, min_distance=0.5 * np.sqrt(avg_piece_area_target))

    print(f"Writing {args.out} ...")
    vis = render_solution(photo, pieces, matches, max_arrows=args.max_arrows,
                           conflicts=conflicts)
    cv2.imwrite(args.out, vis)

    if args.batch_size:
        stem, ext = os.path.splitext(args.out)
        batches = render_batches(photo, pieces, matches, batch_size=args.batch_size,
                                  conflicts=conflicts)
        for i, batch_img in enumerate(batches, start=1):
            batch_path = f"{stem}_batch{i:02d}{ext}"
            cv2.imwrite(batch_path, batch_img)
        print(f"Writing {len(batches)} batch image(s): {stem}_batch01{ext} ...")

    if args.csv:
        write_csv(args.csv, pieces, matches, conflicts=conflicts)
        print(f"Writing {args.csv} ...")

    scores = sorted((m.score for m in matches), reverse=True)
    if scores:
        n = len(scores)
        print(f"Score summary: best={scores[0]:.2f} median={scores[n // 2]:.2f} "
              f"worst={scores[-1]:.2f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
