"""Run the Photon or analytical Proton CT/MR release configuration."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def configure(image_type, device, modality="photon"):
    manifest = json.loads((ROOT / "release-manifest.json").read_text())
    environment = json.loads(
        (ROOT / "configs" / f"{modality}-{image_type}.json").read_text()
    )
    for name in list(os.environ):
        if name.startswith("PRISMX_"):
            del os.environ[name]
    os.environ.update(environment)
    os.environ.update(
        {
            "PRISMX_MODALITY": modality,
            "PRISMX_IMAGE_TYPE": image_type,
            "PRISMX_DEVICE": device,
            "PRISMX_SKIP_WEIGHT_VERIFY": "0",
            "PRISMX_GIT_REV": "v1.0.0",
            "PRISMX_EXPECT_ENGINE_SHA256": manifest["files"][f"weights/{modality}.pt"],
            "PRISMX_EXPECT_MR_NET_SHA256": manifest["files"]["weights/mr-density.pt"]
            if image_type == "mri"
            else "none",
            "PRISMX_EXPECT_COMMISSIONING_SHA256": manifest["commissioning_sha256"],
        }
    )
    for relative, expected in manifest["files"].items():
        if relative in ("weights/photon.pt", "weights/proton.pt") and relative != f"weights/{modality}.pt":
            continue
        if relative == "weights/mr-density.pt" and image_type == "ct":
            continue
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing release file: {relative}. Extract the weight archive here."
            )
        if sha(path) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("photon", "proton"), default=os.environ.get("PRISMX_MODALITY", "photon"))
    parser.add_argument(
        "--image-type",
        choices=("ct", "mri"),
        default=os.environ.get("PRISMX_IMAGE_TYPE", "ct"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--serve", action="store_true", help="Serve /health and /invoke on port 4743"
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if (
        not args.serve
        and not args.verify_only
        and (args.input is None or args.output is None)
    ):
        parser.error("Provide --input and --output, or use --serve / --verify-only")
    configure(args.image_type, args.device, args.modality)
    if args.verify_only:
        print(f"{args.modality} {args.image_type}: release checksums verified")
        return
    sys.path.insert(0, str(ROOT / "scripts"))
    if args.serve:
        if args.input:
            os.environ["INPUT_PATH"] = str(args.input.resolve())
        if args.output:
            os.environ["OUTPUT_PATH"] = str(args.output.resolve())
        from scripts.gc_server import main as serve

        serve()
    else:
        from prismx.runtime import Models, _default_cfg, run

        models = Models(
            args.modality,
            args.image_type,
            args.device,
            _default_cfg(args.modality, args.image_type),
        )
        run(args.input, args.output, models)


if __name__ == "__main__":
    main()
