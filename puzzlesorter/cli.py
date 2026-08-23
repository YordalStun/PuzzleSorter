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

from .align import align_target_to_photo, local_affine
from .board import find_assembled_region
from .pieces import build_roi_mask, detect_pieces
from .match import match_all, flag_conflicts, build_valid_mask
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

    print("Detecting loose pieces...")
    roi = build_roi_mask(photo.shape, board_mask=board.mask, exclude_rects=args.exclude,
                          margin_top=args.margin_top, margin_bottom=args.margin_bottom,
                          margin_left=args.margin_left, margin_right=args.margin_right)
    pieces = detect_pieces(photo, roi, expected_piece_area_photo)
    print(f"  found {len(pieces)} candidate pieces")
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

    print(f"Matching {len(pieces)} pieces against the target image "
          f"(this can take a while)...")
    matches = match_all(pieces, photo, target, aln, search_rect,
                         approx_target_point=board_center_target, valid_mask=valid_mask)
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
