"""Score stored release outputs against identically thresholded MC segments.

CPU only. Does not run or modify a model. Reference dose never selects model
queries. Run after prismx.report_evaluation has saved each complete plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pymedphys
import SimpleITK as sitk


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_segment(dose, cutoff):
    return np.where(dose > np.float32(cutoff), dose, 0.)


def range_metrics(frames):
    ranges = [f["range"]["delta_r80_mm"] for f in frames
              if f.get("range", {}).get("valid")]
    return {"r80_valid": len(ranges),
            "r80_attempted": sum("range" in f for f in frames),
            "r80_mae_mm": float(np.mean(np.abs(ranges))) if ranges else None,
            "r80_bias_mm": float(np.mean(ranges)) if ranges else None}


def metrics(pred, ref, spacing, settings):
    peak = float(ref.max())
    assert peak > 0 and pred.shape == ref.shape
    bands = {"high": ref >= 0.8 * peak,
             "mid": (ref >= 0.3 * peak) & (ref < 0.8 * peak),
             "low": (ref >= 0.1 * peak) & (ref < 0.3 * peak)}
    mae = {key: float(np.abs(pred[mask] - ref[mask]).mean() / peak)
           for key, mask in bands.items()}
    axes = tuple(np.arange(n, dtype=np.float64) * step
                 for n, step in zip(ref.shape, spacing[::-1]))
    gamma = pymedphys.gamma(
        axes, ref, axes, pred, dose_percent_threshold=settings["dose_percent"],
        distance_mm_threshold=settings["distance_mm"],
        lower_percent_dose_cutoff=settings["lower_percent_dose_cutoff"],
        local_gamma=settings["local"], interp_fraction=settings["interp_fraction"],
        max_gamma=settings["max_gamma"],
        ram_available=3 * 10**9)
    valid = ~np.isnan(gamma)  # Infinite values are failures, not exclusions.
    return {"mae_bands": mae,
            "gamma_fraction": float((gamma[valid] <= 1).mean()),
            "gamma_voxels": int(valid.sum()),
            "integral_ratio": float(pred.sum() / ref.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--particle", choices=("photon", "proton"), required=True)
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).with_name("matched-output-evaluation.json"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    root = args.source / f"{args.particle}_ct-{args.case}"
    build = json.loads((root / "build.json").read_text())
    protocol = json.loads(args.protocol.read_text())
    build_protocol = args.protocol.parent / protocol["original_build_protocol"]
    expected = protocol["original_build_protocol_sha256"]
    assert sha(build_protocol) == expected == build["protocol_sha256"], "Build protocol differs"
    assert protocol["reference_cutoff_gy"] == protocol["minimum_cutoff_gy"] == 3e-7
    assert args.case in protocol["cases"]
    assert build["complete_plan"]
    records = json.loads((root / "segments.json").read_text())
    stored = np.load(root / "plan.npz", allow_pickle=False)
    reference = np.zeros_like(stored["ref"], dtype=np.float64)
    matched = np.zeros_like(reference)
    assert len(records) == build["segments"] == len(build["frames"])
    cutoff = protocol["reference_cutoff_gy"]
    for index, record in enumerate(records):
        assert sha(record["gt"]) == build["frames"][index]["gt_file_sha256"]
        dose = sitk.GetArrayFromImage(sitk.ReadImage(record["gt"])).astype(np.float32)
        reference += dose
        matched += threshold_segment(dose, cutoff)
        if (index + 1) % 120 == 0:
            print(args.case, index + 1, len(records), flush=True)
    assert np.array_equal(reference, stored["ref"])
    result = {"case": args.case, "particle": args.particle,
              "release_commit": protocol["release_commit"],
              "cutoff_gy": cutoff, "cutoff_applied": "each segment, before summation",
              "reference_reaccumulation_bit_equal": True,
              "reference_integral_removed": float(1 - matched.sum() / reference.sum()),
              "source_protocol_sha256": build["protocol_sha256"],
              "scoring_protocol_sha256": sha(args.protocol),
              "source_summary_sha256": (sha(args.source / "summary.json")
                                         if (args.source / "summary.json").is_file() else None),
              "scorer_sha256": sha(__file__), "pymedphys": pymedphys.__version__,
              "results": []}
    for image_type in ("ct", "mri"):
        task = f"{args.particle}_{image_type}"
        directory = args.source / f"{task}-{args.case}"
        plan = np.load(directory / "plan.npz", allow_pickle=False)
        task_build = json.loads((directory / "build.json").read_text())
        assert task_build["complete_plan"] and task_build["segments"] == len(records)
        assert task_build["protocol_sha256"] == expected
        assert sha(directory / "plan.npz") == task_build["plan_sha256"]
        assert np.array_equal(plan["ref"], reference)
        row = {"task": task, "case": args.case, "segments": len(records),
               "plan_sha256": sha(directory / "plan.npz"),
               "build_sha256": sha(directory / "build.json"),
               **metrics(plan["pred"], matched, plan["spacing"], protocol["gamma"]),
               "r80_reference": "unthresholded MC; unchanged original range analysis",
               **range_metrics(task_build["frames"])}
        result["results"].append(row)
        (args.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(task, args.case, row["gamma_fraction"], flush=True)


if __name__ == "__main__":
    main()
