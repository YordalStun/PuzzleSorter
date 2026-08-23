import numpy as np
import cv2

from puzzlesorter.color import estimate_color_correction, mean_lab, color_agreement


def test_estimate_color_correction_recovers_known_offset():
    rng = np.random.default_rng(0)
    warped_target = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
    lab = cv2.cvtColor(warped_target, cv2.COLOR_BGR2LAB).astype(np.int16)
    known_offset = np.array([10, -15, 5])
    shifted_lab = np.clip(lab + known_offset, 0, 255).astype(np.uint8)
    photo = cv2.cvtColor(shifted_lab, cv2.COLOR_LAB2BGR)

    mask = np.full((100, 100), 255, np.uint8)
    correction = estimate_color_correction(photo, warped_target, mask)
    assert np.allclose(correction, known_offset, atol=1.5)


def test_color_agreement_decreases_with_distance():
    piece_lab = np.array([50.0, 20.0, 20.0])
    close = np.array([51.0, 19.0, 21.0])
    far = np.array([50.0, 60.0, -40.0])

    agree_close, dist_close = color_agreement(piece_lab, close, correction=np.zeros(3))
    agree_far, dist_far = color_agreement(piece_lab, far, correction=np.zeros(3))

    assert dist_far > dist_close
    assert agree_far < agree_close
    assert 0.0 <= agree_far <= 1.0
    assert 0.9 <= agree_close <= 1.0


def test_color_agreement_uses_correction():
    piece_lab = np.array([50.0, 20.0, 20.0])
    target_patch_lab = np.array([40.0, 20.0, 20.0])  # 10 units off in L only

    agree_uncorrected, dist_uncorrected = color_agreement(piece_lab, target_patch_lab,
                                                            correction=np.zeros(3))
    agree_corrected, dist_corrected = color_agreement(piece_lab, target_patch_lab,
                                                        correction=np.array([10.0, 0.0, 0.0]))

    assert dist_corrected < dist_uncorrected
    assert agree_corrected > agree_uncorrected
    assert dist_corrected < 0.5


def test_mean_lab_matches_manual_conversion():
    img = np.full((10, 10, 3), (100, 150, 200), np.uint8)
    mask = np.full((10, 10), 255, np.uint8)
    mask[:5, :] = 0  # only bottom half counted

    result = mean_lab(img, mask)
    expected = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[5:, :].astype(np.float32).reshape(-1, 3).mean(axis=0)
    assert np.allclose(result, expected, atol=0.5)
