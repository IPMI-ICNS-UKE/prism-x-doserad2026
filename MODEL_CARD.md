# Models

| File | Role | Parameters |
|---|---|---:|
| `weights/photon.pt` | Photon CT/MR dose engine | 96,452 |
| `weights/proton.pt` | Analytical Proton CT/MR dose engine with learned correction | 95,268 |
| `weights/mr-density.pt` | Shared MR-to-density network | 51,045,473 |

Checksums and checkpoint steps are in [weights/manifest.json](weights/manifest.json). Export removes optimizer state and training bookkeeping; model tensors are unchanged.

Photon combines a commissioned depth-dose/fluence prior with TCN context and residual point decoders. Proton uses a commissioned Bragg depth profile and lateral Gaussian with the same correction architecture. Its checkpoint is paired with the supplied Bragg table and lateral-spread parameters; replacing those changes the model.

CT is converted to density using the released mapping. MR uses the supplied density network before the corresponding dose engine. Proton uses the same dose scale for CT and MR. Photon has the released per-site MR scales.

Training uses DoseRAD images and Geant4 reference doses. Dose-engine recipes and validation exclusions are included. The MR network is supplied pretrained, with its training implementation but no reconstructed historical training recipe.

This release covers the analytical–neural method, not every final competition configuration. Photon follows the archived CT/MR reconstruction. Proton outputs are queried within a 40 mm beam-axis radius and the internal grid, masked to a density-derived filled body region, and cutoff-applied. Its output support uses no reference dose. These outputs are not claimed to reproduce the report's reference-dose-defined evaluation support or final leaderboard scores.

The historical geometry assumes identity image direction and the DoseRAD beam conventions. Arbitrary clinical plans, other scanners and coordinate conventions are not validated. No clinical validation is claimed.
