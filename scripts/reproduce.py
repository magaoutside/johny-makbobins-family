"""Run and verify the published V52/V58 frozen replay packages."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check_assets():
    manifest = json.loads((ROOT / "artifacts.sha256.json").read_text())
    for name, expected in manifest.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    return len(manifest)


def check_submission(actual_path, expected_path, exact):
    import numpy as np
    import pandas as pd

    actual = pd.read_csv(actual_path, dtype={"id": str})
    expected = pd.read_csv(expected_path, dtype={"id": str})
    if list(actual.columns) != ["id", "score"]:
        raise ValueError("Submission columns must be id,score")
    if len(actual) != 362 or not actual.id.is_unique or actual.id.isna().any():
        raise ValueError("Expected 362 unique test ids")
    if actual.id.tolist() != expected.id.tolist():
        raise ValueError("Test id order mismatch")
    if not np.isfinite(actual.score).all() or not actual.score.between(0, 1).all():
        raise ValueError("Scores must be finite values in [0,1]")
    maximum_error = float(np.max(np.abs(actual.score - expected.score)))
    if maximum_error > 1e-10:
        raise ValueError(f"Submission score mismatch: {maximum_error}")
    identical_bytes = actual_path.read_bytes() == expected_path.read_bytes()
    if exact and not identical_bytes:
        raise ValueError("V58 canonical CSV differs byte-for-byte")
    return {"rows": len(actual), "maximum_absolute_error": maximum_error,
            "byte_identical": identical_bytes,
            "sha256": hashlib.sha256(actual_path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["all", "v52", "v58"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    count = check_assets()
    versions = ["v52", "v58"] if args.version == "all" else [args.version]
    result = {"verified_artifacts": count, "versions": {}}
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    for version in versions:
        bundle = ROOT / "solutions" / version
        output = args.output.resolve() / version
        subprocess.run([sys.executable, str(bundle / "replay.py"),
                        "--bundle", str(bundle), "--output", str(output)], check=True, env=env)
        filename = "test_a20.csv" if version == "v52" else "submission.csv"
        result["versions"][version] = check_submission(
            output / filename, ROOT / "submissions" / f"{version}.csv", exact=version == "v58"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
