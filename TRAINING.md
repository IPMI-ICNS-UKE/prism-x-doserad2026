# Training

Obtain the CT/MR images, plans and beam-level Geant4 doses through [DoseRAD](https://doserad2026.grand-challenge.org/data/) under its terms. No patient data are redistributed here. No additional synthetic training dataset was used for the released models.

Expected case layout:

```text
photon/training/<case-id>/
  <case-id>.json
  image/ct.mha
  image/mr.mha
  dose/Dose_B0_CP000.mha
  ...
```

The Photon recipe is in `configs/training/photon.json`; its validation exclusions are in `configs/splits.json`. Preprocessing is implemented in `prismx/hu.py`, `mr_data.py` and `plan_data.py`.

## Photon engine

```sh
python train.py photon --data /data/photon/training --out /results/photon
```

Commands are printed first; add `--execute` to train. The historical run used 60,000 updates, 8,192 query points/update, AdamW, EMA and checkpoints every 500 updates. The submitted weights are the EMA at update 60,000, not necessarily `best.pt`.

## MR density network

Photon MR and Proton MR use the supplied pretrained checkpoint `weights/mr-density.pt`.
The network definition and training code are included in `prismx/mr_density.py` and `scripts/train_mr_density.py`. A historical MR retraining recipe is not provided.

The trainer supports density supervision and an optional dose-supervised stage; this is not a claim that the supplied checkpoint used both. A small synthetic density-only training check is:

```sh
python -m scripts.train_mr_density --smoke --device cpu \
  --anchor-steps 2 --dose-steps 0 --out /tmp/prismx-mr-smoke
```

Use a new output directory. For the data-backed training options, run `python -m scripts.train_mr_density --help`. The smoke test is not a historical recipe or an accuracy experiment.

## Proton engine

```sh
python train.py proton --data /data/proton/training --out /results/proton
```

Add `--execute` to start training. Proton cases use the same layout, with dose files named `Dose_B{beam}_R{ray}_L{layer}.mha`. The recipe in `configs/training/proton.json` uses 50,000 updates, 8,192 queries/update, AdamW, EMA and an on-axis depth-dose loss. Validation patients are listed in `configs/splits.json`.

The trainer preserves the checkpoint's analytical basis: entrance width from `proton_bragg.npz`, depth-growth parameters `(0.02, 1.5)`, and no extra halo. The recipe's training dose scale and the table's inference calibration differ slightly; each is used at its respective stage. Training starts from initialization, not from the released weights.

The supplied Photon checkpoint is the 60,000-update model; Proton is the 50,000-update model. These historical releases are not the newer paper retrainings. Exact numeric reproduction across hardware and library versions is not guaranteed.
