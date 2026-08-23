# PuzzleSorter

Points you to where each loose jigsaw piece goes, from two photos:

1. **`--unsolved`** — a photo of the puzzle as it currently sits: the
   assembled-so-far portion plus the scattered loose pieces around it.
2. **`--target`** — a reference photo of the finished picture (e.g. the puzzle
   box). It doesn't need to be a clean scan — a photo taken at an angle,
   held in your hand, works fine.

Output is an annotated copy of your `--unsolved` photo: each loose piece gets
a number, and an arrow points from the piece to the spot on the board where
it's predicted to belong.

## How it works

1. **Align**: ORB feature matching + RANSAC finds a homography between the
   target image and the photo, using whatever's already assembled as the
   shared content. This tells us exactly how any point in the target picture
   maps onto the physical board in your photo — including scale and rotation
   differences from the two photos being taken at different angles.
2. **Locate the assembled region**: a texture/gradient-energy blob detector
   finds the already-built portion of the puzzle in the photo, so it can be
   excluded from piece search and used to estimate a single piece's pixel
   size (from the target picture's area divided by `--pieces`).
3. **Find loose pieces**: the same texture-energy signal, restricted to
   everywhere *outside* the assembled region and any `--exclude` zones,
   segments loose pieces. Piles of touching pieces are split with a
   watershed transform seeded from the expected single-piece size.
4. **Classify each piece's edge shape**: a piece's segmented silhouette is
   fit against its nominal square/rectangular footprint to classify each of
   its 4 sides as straight, tab (bump out) or blank (notch in). This is a
   geometric fact independent of the piece's printed content: a straight
   edge is only physically possible on the picture's outer border, and two
   straight edges (adjacent, on a real piece) only at a corner. That's used
   later to hard-constrain where the piece is even allowed to match, not
   just as another content signal.
5. **Find gaps**: `find_assembled_region` only knows the assembled block's
   outer extent, not which spots *within* it are already filled — a piece
   not yet placed leaves a gap that's invisible to a simple outer contour.
   The same texture-energy signal, now applied inside the block, tells gaps
   (plain mat, low texture) from filled spots (any printed piece content).
   Matches are restricted to actual gaps, so a piece is never pointed at
   somewhere already occupied by a different, correctly-placed piece.
6. **Match**: each piece is matched against the target image with a
   coarse-to-fine search over rotation (0-360°) and position, using masked
   normalized cross-correlation (only the piece's own pixels are compared,
   not its background) combined with an explicit color-agreement check
   (calibrated against the assembled region, to correct for the two photos'
   different lighting/white balance) — correlation alone is tolerant of
   outright hue mismatches, so without this a green piece can outscore a
   blue one that's actually a better structural fit. Matches are also
   constrained to stay inside the target picture's actual (possibly tilted)
   boundary — the axis-aligned box drawn around a tilted picture has corners
   that fall outside it, e.g. onto a box's cardboard border or glare, which
   otherwise can win spuriously confident matches against low-detail pieces.
   A piece classified as a border or corner piece (step 4) is additionally
   confined to a band hugging the picture's edge, or one of its 4 corners —
   so a border piece never gets pointed at the picture's interior just
   because its content happens to correlate well there. The match location
   is projected back onto the photo via the homography from step 1 to get
   the arrow's destination.
7. **Render**: numbered labels on every piece and its destination, arrows
   drawn for the highest-confidence matches (`--max-arrows`), color-coded by
   match confidence (green/orange/red). `--batch-size` additionally splits
   the arrows across several smaller, much less cluttered images.

## Install

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
python -m puzzlesorter.cli --unsolved photo.jpg --target box.jpg --pieces 500 --out solution.jpg --csv solution.csv
```

Real photos usually have clutter on the table (tools, boxes, glasses, a cup —
anything that isn't the mat/cloth or a puzzle piece). Exclude those regions
by pixel rectangle so they aren't mistaken for pieces:

```
--exclude 230,420,330,140 --exclude 1750,0,248,580
```

### Solving in multiple rounds without a new photo each time

`--iterate ROUNDS` simulates placing the high-confidence pieces from one
round and re-solving for what's left, entirely from the photos you already
gave it — no need to place pieces for real and re-photograph between
rounds:

```
python -m puzzlesorter.cli --unsolved photo.jpg --target box.jpg --pieces 500 \
  --out solution.jpg --csv solution.csv --iterate 5
```

This is a *simulation*: it doesn't know whether you actually placed the
suggested pieces, it just assumes each round's placements are correct and
builds the next round on top of that assumption. Wrong assumptions compound,
so it only auto-commits a placement when its score clears
`--auto-place-threshold` (default 0.45, stricter than the 0.40 "high
confidence" display cutoff elsewhere) — everything else is left for a plain
`--batch-size` run instead, where a human looks at each suggestion before
acting on it.

Output: `solution_round01.jpg`, `solution_round02.jpg`, ... — one image per
round showing only that round's newly-placed pieces (arrows from where they
were to where they went) — plus `solution_final.jpg`, the cumulative working
image after every round, and one CSV covering every piece placed across all
rounds. Real, physical pieces still need to move to match this - it's telling
you what to do across several rounds at once, not doing it for you.

Other useful flags:
- `--pieces N` — the puzzle's piece count (on the box). Only used to estimate
  one piece's pixel size; doesn't need to be exact.
- `--margin-top/bottom/left/right N` — crop out photo edges (chair, wall,
  window, etc.) from the piece search.
- `--max-arrows N` — cap how many arrows are drawn on the single `--out`
  overview image so it stays legible; every piece still gets a numbered
  label either way.
- `--batch-size N` — also write a series of clearer images with only N
  arrows each (`solution_batch01.jpg`, `solution_batch02.jpg`, ...),
  highest-confidence first. Recommended for puzzles with more than a
  couple dozen loose pieces — a single image with 50+ crossing arrows is
  unreadable. 8-12 is a good batch size.
- `--min-inliers N` — raise this if alignment succeeds on a coincidental
  handful of feature matches.

## Limitations

- **Touching/overlapping pieces**: watershed splitting isn't perfect. A
  cluster that doesn't get separated is matched as one blob, which usually
  produces a low confidence score (visible as a red/orange label) rather
  than a confident wrong answer — but for the best results, spread pieces
  out so they aren't touching before photographing.
- **Low-detail pieces** (plain sky, water, single-color areas) are
  inherently ambiguous from image content alone — same as for a human
  solver. Trust the confidence color coding.
- **Gap detection** is texture-based, same as loose-piece detection: a very
  small single-piece gap right at the edge of the assembled block, or right
  next to a low-texture (plain sky/water) piece, can be missed or merged
  into the "filled" area, making that spot unavailable as a match target.
- **Conflicting claims**: if two pieces' best match lands on essentially the
  same spot, the lower-scoring one is marked with a `?` and gets no arrow
  (it's shown but not trusted) rather than risking two confident-looking
  arrows pointing at the same place.
- **Edge-shape classification** depends on a clean single-piece silhouette:
  a still-merged watershed cluster, an unusually shallow tab/notch, or messy
  segmentation can misclassify a side in either direction — reading a real
  straight edge as a shallow tab (losing a constraint that would've helped),
  or, less often, reading a genuinely interior edge as straight (wrongly
  confining that piece to the border/corner search). A piece whose
  classification fails outright, or comes back with no straight edges at
  all, gets no border/corner constraint rather than a wrong one — but a
  confident misclassification is still possible and shows up as a piece
  that never finds a good match despite obviously belonging elsewhere.
- **Runtime** scales with piece count × search resolution; a few hundred
  loose pieces can take several minutes.
- Clutter on the table (tools, boxes, cups) needs to be pointed out via
  `--exclude`/`--margin-*` — it isn't detected automatically.
- **`--iterate` compounds errors**: each round trusts every prior round's
  placements as ground truth. A wrong auto-placement doesn't just mislabel
  one piece - it can throw off gap detection and color calibration for every
  piece considered afterward. The auto-place threshold is deliberately
  strict to make this rare, but it isn't zero; check each round's image
  against your real board before moving to the next.
