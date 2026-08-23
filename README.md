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
4. **Match**: each piece is matched against the target image with a
   coarse-to-fine search over rotation (0-360°) and position, using masked
   normalized cross-correlation (only the piece's own pixels are compared,
   not its background). The match location is projected back onto the photo
   via the homography from step 1 to get the arrow's destination.
5. **Render**: numbered labels on every piece and its destination, arrows
   drawn for the highest-confidence matches (`--max-arrows`), color-coded by
   match confidence (green/orange/red).

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

Other useful flags:
- `--pieces N` — the puzzle's piece count (on the box). Only used to estimate
  one piece's pixel size; doesn't need to be exact.
- `--margin-top/bottom/left/right N` — crop out photo edges (chair, wall,
  window, etc.) from the piece search.
- `--max-arrows N` — cap how many arrows are drawn so the image stays
  legible; every piece still gets a numbered label either way.
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
- **Runtime** scales with piece count × search resolution; a few hundred
  loose pieces can take several minutes.
- Clutter on the table (tools, boxes, cups) needs to be pointed out via
  `--exclude`/`--margin-*` — it isn't detected automatically.
