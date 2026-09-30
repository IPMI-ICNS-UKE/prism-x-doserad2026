# Data and licensing

The code, pretrained weights and commissioned tables in this release are licensed under [Apache-2.0](LICENSE).

Patient images, plans and reference dose maps are not redistributed. Obtain the [DoseRAD dataset](https://doserad2026.grand-challenge.org/data/) from its providers under CC BY-NC 4.0 and cite the [dataset paper](https://doi.org/10.48550/arXiv.2604.12778). This repository's license does not replace the terms governing access to or use of that dataset.

Dependencies retain their own licenses:

- [PyTorch](https://github.com/pytorch/pytorch/blob/v2.3.1/LICENSE)
- [NumPy](https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt)
- [SciPy](https://github.com/scipy/scipy/blob/v1.14.1/LICENSE.txt)
- [SimpleITK](https://github.com/SimpleITK/SimpleITK/blob/master/LICENSE)

The Docker base includes additional runtime components under their respective terms. No Geant4 code or external dose-prediction model is redistributed.
