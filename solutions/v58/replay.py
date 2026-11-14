"""Reproduce V58 from frozen expert predictions, without training or RGB access."""
from pathlib import Path
import argparse
import hashlib

import numpy as np
import pandas as pd

EXPECTED_SHA256 = "1ccc86a345fed7590f0462a2216aa9f07f032df0c68a63decf6cdcde2d8087d6"


def assemble_scores(data):
    """Apply the frozen class-specific corrections in the original order."""
    baseline = data["baseline"]
    classes = data["class_name"]
    n_oof = len(data["new_dense_oof_iou"])

    # The full-image expert may raise a score; the crop expert may lower it.
    scores = baseline + np.clip(0.2 * (data["p_full"] - baseline), 0, 0.1)
    scores += np.clip(0.2 * (data["p_crop"] - baseline), -0.1, 0)
    hands = classes == "nude_hands"
    scores[hands] = baseline[hands] + np.clip(
        0.2 * (data["p_anchor_adapted"][hands] - baseline[hands]), -0.1, 0.1
    )
    scores = np.clip(scores, 0, 1)

    # Build the replacement dense component relative to V22.
    oof_iou = data["new_dense_oof_iou"]
    test_iou = data["new_dense_test_iou"]
    v22 = data["v22"]
    dense = v22.copy()
    valid_oof = np.isfinite(oof_iou)
    oof_target = np.clip((0.85 - oof_iou) / 0.5, 0, 1)
    dense[:n_oof][valid_oof] += 0.25 * (oof_target - v22[:n_oof])[valid_oof]

    # Both fit-day experts must have geometry and agree on correction direction.
    test_target = np.clip((0.85 - test_iou) / 0.5, 0, 1)
    delta = test_target - v22[n_oof:][None, :]
    agree_up = (delta >= 0).all(axis=0)
    agree_down = (delta <= 0).all(axis=0)
    supported = np.isfinite(test_target).all(axis=0) & (agree_up | agree_down)
    safe = np.where(np.isfinite(test_target), test_target, v22[n_oof:][None, :])
    conservative = np.where(agree_up, safe.min(axis=0), safe.max(axis=0))
    dense[n_oof:][supported] += 0.25 * (conservative - v22[n_oof:])[supported]

    # Preserve the previous dense contribution when new evidence is unsupported.
    cups = (classes == "cup") & np.r_[valid_oof, supported]
    scores[cups] += (dense - data["v31"])[cups]

    trays = classes == "tray"
    probability = np.clip(baseline[trays], 0.0001, 0.9999)
    scores[trays] = 1 / (1 + np.exp(-(np.log(probability / (1 - probability)) - 0.6)))
    scores = np.clip(scores, 0, 1)
    return baseline + np.clip(scores - baseline, -0.15, 0.15)


def replay(bundle, output):
    with np.load(bundle / "replay_inputs.npz", allow_pickle=False) as data:
        scores = assemble_scores(data)
        max_error = float(np.max(np.abs(scores - data["expected"])))
        if not np.allclose(scores, data["expected"], rtol=0, atol=1e-14):
            raise ValueError(f"Frozen score mismatch: {max_error}")
        if not np.array_equal(np.round(scores, 12), np.round(data["expected"], 12)):
            raise ValueError("Canonical 12-decimal score mismatch")
        n_oof = len(data["new_dense_oof_iou"])
        submission = pd.DataFrame({"id": data["id"][n_oof:], "score": scores[n_oof:]})
    output.mkdir(parents=True, exist_ok=True)
    path = output / "submission.csv"
    submission.to_csv(path, index=False, float_format="%.12f")
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != EXPECTED_SHA256:
        raise ValueError(f"Submission SHA256 mismatch: {actual_sha}")
    print(f"V58 EXACT_REPLAY_OK | {len(scores)} scores | SHA256 {actual_sha}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("outputs/v58"))
    args = parser.parse_args()
    replay(args.bundle, args.output)
