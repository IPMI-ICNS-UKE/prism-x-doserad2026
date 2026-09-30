"""Print the selected dose-engine training command; --execute runs it."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recipe",
        choices=("photon", "proton"),
    )
    parser.add_argument(
        "--data", required=True, type=Path, help="Original training-case root for the selected modality"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    recipe = json.loads((ROOT / "configs/training" / f"{args.recipe}.json").read_text())
    data, out = args.data.resolve(), args.out.resolve()
    if not data.is_dir():
        parser.error("--data must be an existing case directory root")
    values = dict(recipe["args"])
    values.update(
        {
            "data": str(data),
            "out": str(out),
            "device": args.device,
        }
    )
    table_key = "pdd" if args.recipe == "photon" else "bragg"
    table_name = "synthrad_photon_pdd.npz" if args.recipe == "photon" else "proton_bragg.npz"
    values[table_key] = str(ROOT / "commissioning" / table_name)
    module = recipe["script"].removesuffix(".py").replace("/", ".")
    command = [sys.executable, "-m", module]
    for key, value in values.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        command.append(flag)
        if value is not True:
            command.append(str(value))
    print(shlex.join(command), flush=True)
    if not args.execute:
        return
    if out.exists() and any(out.iterdir()):
        parser.error("--out must be empty; this launcher does not resume runs")
    out.mkdir(parents=True, exist_ok=True)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("PRISMX_")}
    environment.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
