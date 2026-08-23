import numpy as np
import cv2

from puzzlesorter.align import align_target_to_photo, local_affine


def _synthetic_textured_image(size=900, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    for _ in range(500):
        pt1 = tuple(rng.integers(0, size, size=2).tolist())
        pt2 = tuple((np.array(pt1) + rng.integers(-80, 80, size=2)).tolist())
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        if rng.random() < 0.5:
            cv2.line(img, pt1, pt2, color, int(rng.integers(1, 6)))
        else:
            cv2.circle(img, pt1, int(rng.integers(5, 30)), color, -1)
    return img


def test_align_recovers_known_similarity_transform():
    target = _synthetic_textured_image(size=500, seed=3)

    true_scale = 1.6
    true_angle = 12.0
    true_tx, true_ty = 300, 150
    M = cv2.getRotationMatrix2D((250, 250), true_angle, true_scale)
    M[0, 2] += true_tx
    M[1, 2] += true_ty

    photo = np.zeros((1000, 1200, 3), np.uint8)
    warped = cv2.warpAffine(target, M, (1200, 1000))
    photo[:] = warped

    aln = align_target_to_photo(target, photo, min_inliers=10)
    assert aln.inlier_matches >= 10

    # a target-space point warped through the recovered homography should land
    # close to where the known affine transform puts it
    for pt in [(100, 100), (400, 50), (250, 400)]:
        expected = M @ np.array([pt[0], pt[1], 1.0])
        got = cv2.perspectiveTransform(np.float32([[pt]]), aln.homography)[0, 0]
        assert np.hypot(got[0] - expected[0], got[1] - expected[1]) < 5.0

    scale, rotation = local_affine(aln, (250, 250))
    assert abs(scale - true_scale) < 0.05
    # cv2's rotation matrices use a clockwise-positive-y-down convention; just
    # check magnitude/sign roughly agree rather than pin an exact sign convention
    assert abs(abs(rotation) - true_angle) < 2.0
