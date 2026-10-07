# Verification

The release is checked separately from the active research repository.

- Source projection and exported model tensor identities are checked against the archived reconstruction.
- Source-only CPU tests require no patient data or pretrained weights.
- `scripts/check_install.py` verifies the supplied Photon and MR models on synthetic CPU inputs.
- Photon CT/MR container builds and `/health` plus an empty `/invoke` passed with PyTorch 2.3.1 / CUDA 12.1.

## Real-input regression, 27 September 2026

On an NVIDIA A40 with PyTorch 2.3.1 / CUDA 12.1, the release containers were compared with preserved output from the archived-source reconstruction:

- Two existing development patients: one abdominal, one thoracic.
- Twelve control points per patient for both CT and MR: 48 dose maps in total.
- All 48 stored dose arrays were bit-identical to the reconstruction reference.
- Output geometry, cutoff and all ten output slots passed.

These patients are not an independent test cohort for the historical models. The check verifies packaging, not generalization or equivalence to a recovered original submission image.

## Training and source checks

- Photon: two fresh training updates on real CT data, validation and checkpoint reload passed. Saved EMA tensors were finite and optimizer state showed updates.
- MR: two density-supervised updates on a synthetic water box, validation and checkpoint reload passed. No dose fine-tuning or historical recipe reconstruction was performed.
- Six source-only tests passed: Photon and MR forward/backward, cutoff boundary, task scope, patient split, and command-line entry points.

## Proton addition, 30 September 2026

- The retained analytical Proton checkpoint contains 95,268 parameters at update 50,000. Exported EMA tensors match the original exactly.
- Six CPU phantom configurations (water/tissue interface, three energy/angle pairs; 1,536 point queries) are bit-identical to the archived manuscript code.
- The Proton trainer completed two synthetic updates with finite weights, a parameter change and a readable optimizer checkpoint.
- Synthetic CT inference passed image-level isocenter routing, geometry preservation, output cutoff and ten-slot serialization checks.
- Nine source-only tests passed. Existing Photon computational modules and Photon/MR-density weights are unchanged; only the task dispatcher was extended.

The new Proton adapter has not been evaluated on the hidden leaderboard or compared with the final submitted Proton containers. Photon A40 evidence above applies to the earlier Photon package. No full historical retrain or clinical validation is claimed.

## Report evaluation, 7 October 2026

All eight patient-task plans were evaluated with the unchanged v1.0.0 weights and calibration. The revised tables use matched per-segment cutoffs and the archived scoring protocol in `experiments/report/`. The exact R80 selection retains valid and excluded beamlets.

Eight CPU checks passed for the report metric functions, cutoff-before-summation, gamma failure handling, protocol/table hashes and agreement of all individual R80 measurements with the published aggregates. These checks do not constitute a second full inference run.
