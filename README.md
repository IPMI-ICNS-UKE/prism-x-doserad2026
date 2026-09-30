# PRISM-X / PRISM-P

Compact analytical–neural dose engines for Photon and Proton CT/MR, developed for DoseRAD 2026. An analytical baseline is corrected by a depth-wise TCN and point-wise MLPs. MR images are first converted to density by a shared pretrained network.

Includes inference, training code, dose-engine recipes, calibrated tables and pretrained weights. No patient data or later paper experiments are included. Proton uses the analytical baseline, not DoTA.

## Install

Use Linux, Python 3.10 and PyTorch 2.3.1 / CUDA 12.1. Install `requirements.txt`, then extract the weights archive from [Releases](https://github.com/IPMI-ICNS-UKE/prism-x-doserad2026/releases) into this directory.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/check_install.py
```

## Inference

```sh
python run.py --modality photon --image-type ct --input /data/input --output /data/output
python run.py --modality proton --image-type mri --input /data/input --output /data/output
```

Both modalities support `ct` and `mri`. Inputs use the DoseRAD Grand Challenge image/beam-metadata format; outputs are cutoff-applied 4D MetaImage stacks. No reference dose is required at inference.

```sh
docker build --build-arg MODALITY=proton --build-arg IMAGE_TYPE=ct -t prism-p-ct .
```

Containers serve `/health` and `/invoke` on port 4743. Mount inputs read-only at `/input`, an empty output directory at `/output`, and expose a CUDA GPU. The historical geometry assumes identity image direction.

## Training and models

[Training](TRAINING.md) · [Models](MODEL_CARD.md) · [Checks](VERIFICATION.md)

This is a method release, not a reproduction of all final leaderboard containers. The software is not clinically validated.

## License

Code, weights and calibrated tables: [Apache-2.0](LICENSE). The separately obtained DoseRAD dataset retains its own terms; see [Data and licensing](DATA_AND_LICENSE.md).
