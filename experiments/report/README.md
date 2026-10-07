# Report experiments

The revised report uses the released Photon EMA at update 60000, analytical Proton EMA at update 50000 and final shared MR density model on 1ABB118 and 1THB011. Both patients were excluded throughout MR training. They are development cases, not an independent test set, and this evaluation does not use cross-validation.

## Generate outputs

From the release root, install the released weights and evaluation dependencies. The complete Table 2/3 evaluation takes two commands, using new output directories:

```sh
python -m pip install -r experiments/report/requirements.txt
python experiments/report/reproduce.py --mode build --data /path/to/data --outputs /new/output --execute
python experiments/report/reproduce.py --mode score --outputs /new/output --scores /new/scores --execute
```

Without `--execute`, the commands are only printed. Build visits both patients and all four tasks sequentially and requires CUDA. Scoring is CPU-only. No training is performed. Build uses PyTorch 2.3.1/CUDA 12.1. Scoring uses NumPy 1.26.4, SciPy 1.13.1, SimpleITK 2.5.5, PyMedPhys 0.41.0 and Numba 0.61.2. The first serialized output is checked against the captured dose. Model, calibration, source and output checksums are recorded.

The lower-level `prismx.report_evaluation --mode score` compares cutoff-applied predictions to raw MC references. It is not needed by the commands above; its plan metrics are diagnostic only and are not the revised tables.

## Matched output scoring

The revised tables apply the same 3e-7 Gy cutoff to each MC segment before plan summation. Predictions are unchanged. The protocol is matched-output-evaluation.json; release-evaluation.json preserves the original build configuration.

The completed eight case-task results and their source hashes are in [release-results.json](release-results.json). Gamma fractions and normalized MAE values must be multiplied by 100 for the table percentages. The wrapper prints the rounded table entries after scoring and retains full-precision JSON results in each score directory.

```sh
python experiments/report/score_matched_output.py --source /new/output --out /new/scores/photon-1ABB118 --case 1ABB118 --particle photon
```

Each call scores CT and MR for one patient and particle. It reads the completed case directories, checks reference file hashes, verifies bit-identical reference accumulation and writes metrics without rerunning inference. Its R80 values retain the separate unthresholded-reference range analysis.

## Range and commissioning records

[r80-beamlets.json](r80-beamlets.json) lists all 150 selected beamlets per patient and Proton task by segment, beam, ray and layer, including valid and excluded observations. Selection is `numpy.linspace(0, n_segments - 1, 150, dtype=int)` in original plan order. The file contains each measured crossing, signed error and validity flag, plus source checksums. Excluded observations remain in the archive and are not replaced.

[commissioning.json](commissioning.json) identifies the baseline-fitting patient, segment counts and table hashes. Baseline fitting uses 1ABB006; later amplitude calibration also uses the two development cases. Neither is refitted for this evaluation.

## Model pairing

The released Proton trainer uses distal point weight 2 in the relative squared-error term and a central-axis depth-profile loss weight of 0.5. Checkpoint and loss metadata are in proton-checkpoint.json. No weights or calibration files are changed by this evaluation.
