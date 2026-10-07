"""Print or execute the complete Table 2/3 evaluation with released weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "score"), required=True)
    parser.add_argument("--data", type=Path, help="Contains photon/training and proton/training")
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "build" and args.data is None:
        parser.error("--data is required for build")
    if args.mode == "score" and args.scores is None:
        parser.error("--scores is required for score")
    protocol = json.loads((HERE / "matched-output-evaluation.json").read_text())
    outputs = args.outputs.resolve()
    commands = []
    if args.mode == "build":
        for task in protocol["tasks"]:
            for case in protocol["cases"]:
                commands.append([sys.executable, "-m", "prismx.report_evaluation",
                                 "--mode", "build", "--data", str(args.data.resolve()),
                                 "--task", task, "--case", case,
                                 "--out", str(outputs / f"{task}-{case}")])
    else:
        for particle in ("photon", "proton"):
            for case in protocol["cases"]:
                commands.append([sys.executable, str(HERE / "score_matched_output.py"),
                                 "--source", str(outputs), "--out",
                                 str(args.scores.resolve() / f"{particle}-{case}"),
                                 "--particle", particle, "--case", case])
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)
    if args.execute and args.mode == "score":
        rows = []
        for particle in ("photon", "proton"):
            for case in protocol["cases"]:
                path = args.scores / f"{particle}-{case}" / "summary.json"
                rows.extend(json.loads(path.read_text())["results"])
        for row in rows:
            bands = "/".join(f"{100 * row['mae_bands'][key]:.2f}"
                             for key in ("high", "mid", "low"))
            print(f"{row['task']} {row['case']}: MAE {bands} %, "
                  f"gamma {100 * row['gamma_fraction']:.1f} %, "
                  f"integral {row['integral_ratio']:.3f}")


if __name__ == "__main__":
    main()
