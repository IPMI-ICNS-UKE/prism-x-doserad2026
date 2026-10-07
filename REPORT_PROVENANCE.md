# Report evaluation

The revised report evaluates release v1.0.0 (commit 45b999e): Photon EMA 60000, analytical Proton EMA 50000 and the final shared MR density network. Both CT and MR use 1ABB118 and 1THB011, with all 540 Photon segments or 1080 Proton beamlets per patient.

Both patients were excluded throughout MR training, confirmed by the author on 6 October 2026. They remain development cases because they were used during model development. This is not cross-validation.

## Output maps

Predictions come from the unchanged reference-free release exporter. Both predicted and MC reference segment maps use a cutoff of 3e-7 Gy before unit-weight plan accumulation. Reference dose never selects prediction queries. Plan metrics cover the native grid, with dose-band MAE and local 1%/1 mm gamma above 10% of the reference plan maximum. Gamma uses interpolation fraction 10 and maximum gamma 2, counting infinite values as failures.

R80 is a separate scoring analysis against unthresholded MC beamlets, using the same task-derived WED coordinate for prediction and reference. Shared-GPU component timings are not dedicated runtime benchmarks.

## Model pairing

The MR checkpoint is the final shared network, not a fold model. Its 172 EMA tensors match the original final_submit_anchor.pt. Proton MR resamples the input to 2-mm isotropic spacing before normalization and returns density to the native grid.

Photon MR scales are the release values, 5.4415101245787755e-5 Gy for abdomen and 5.4400891126726915e-5 Gy for thorax. No model, scale or preprocessing option was selected from the diagnostic controls for the revised tables.

## Reproduction

The [experiment directory](experiments/report/README.md) contains executable commands for Tables 2 and 3, the scoring configuration, all eight patient-task results and the exact R80 beamlet selection with individual measurements. Its commissioning record distinguishes baseline fitting on 1ABB006 from amplitude calibration on the development cases. The supplied weights suffice for inference. Reproducing the complete multi-stage MR training from random initialization additionally requires the initial transition checkpoint selection and the exact abdomen-only training view; the known stages are recorded in configs/report/mr-lineage.json.

This evaluation uses neither the newer journal-paper engines nor hidden leaderboard results.
