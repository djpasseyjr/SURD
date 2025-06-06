# SURD: Synergistic-Unique-Redundant Decomposition of causality

_A Python repository for decomposing causality into its synergistic, unique, and redundant components for complex and chaotic systems._

## FORK NOTES:

This fork adds package structure to install with pip:

`pip install git+https://github.com/djpasseyjr/SURD.git`

```python
import numpy as np
import surd

agent = np.random.rand(4, 1000)
target = agent[0, :] * agent[1, :] + agent[2, :]
Y = np.vstack([target, agent])
I_R, I_S, MI = surd.surd_hd(Y, nbins=20, max_combs=3)
```

## Introduction
SURD (Synergistic-Unique-Redundant Decomposition) is a causal inference method that measures the increments of information gained about future events based on the
available information from past observations. It further decomposes the causal interactions into redundant, unique, and synergistic contributions according to their nature. The formulation is non-intrusive and requires only pairs of past and future events, facilitating its application in both computational and experimental
investigations. SURD also identifies the amount of causality that remains unaccounted for due to unobserved variables. The approach can be used to detect
causal relationships in systems with multiple variables, dependencies at different time lags, and instantaneous links.

<!---<img width="1209" alt="Screenshot 2024-02-21 at 5 22 41 PM" src="https://github.mit.edu/storage/user/27189/files/77ed0e78-e988-406c-b977-0e927661b168">--->
![github_diagram_2](https://github.com/user-attachments/assets/937fa18d-1179-4cfe-ae5a-e262582614a0)

## Features

- **Quantification of causality**: SURD measures the increments of information gained about future events from past observations.

- **Decomposition of causality**: It decomposes causal interactions into redundant, unique, and synergistic contributions.

- **Quantification of information leak**: SURD quantifies the amount of causality that remains unaddressed due to unobserved variables.

- **Non-intrusive formulation**: The method is designed to be non-intrusive, requiring only pairs of past and future events for analysis. This facilitates its application across both computational and experimental studies.

- **Probabilistic framework**: SURD provides a probabilistic measure of causality, emphasizing transitional probabilities between states.

- **Dependence on data availability**: The accuracy of the method is contingent on the availability of sufficient data, as it involves estimating probability distributions in high-dimensional spaces.

## System requirements

SURD is designed to operate efficiently on standard computing systems. However, the computational demands increase with the complexity of the probability density functions being estimated. To ensure optimal performance, we recommend a minimum of 16 GB of RAM and a quad-core processor with a clock speed of at least 3.3 GHz per core. The performance metrics provided in this repository are based on tests conducted on macOS with an ARM64 architecture and 16 GB of RAM, and on Linux systems running Red Hat version 8.8-0.8. These configurations have demonstrated sufficient performance for the operations utilized by SURD. Users should consider equivalent or superior specifications to achieve similar performance.

## Getting started

After cloning the repository, you can set up the environment needed to run the scripts successfully by following the instructions below. You can create an environment using `conda` with all the required packages by running:
```sh
conda env create -f environment.yml
```
This command creates a new conda environment and installs the packages as specified in the `environment.yml` file in about 50 seconds. After installing the dependencies, make sure to activate the newly created conda environment with:
```sh
conda activate surd
```
Should you wish to use the transport map for estimating probability density functions (refer to this [tutorial](https://github.com/MIT-Computational-Turbulence-Group/SURD/blob/main/examples/E07_transport_map.ipynb)), the `mpart` library is required. Installation can be executed via the following `conda` command:
```sh
conda install -c conda-forge mpart
```
For users operating on the `osx-arm64` platform, please note that the `mpart` package is not available on `conda-forge`. As an alternative, the library should be installed using `pip`:
```sh
pip install mpart
```
For comprehensive details regarding the installation and further information about the library, please visit the [MParT documentation](https://measuretransport.github.io/MParT/).

## Tutorials
SURD has been applied in a large collection of scenarios that have proven challenging for causal inference and demonstrate its application in analyzing the energy cascade in isotropic turbulence. For examples, consult the documentation or see the Jupyter notebooks in the examples folder.

## Citation

If you use SURD in your research or software, please cite the following paper:

```bibtex
@article{surd,
author={Mart{\'\i}nez-S{\'a}nchez, {\'A}lvaro and Arranz, Gonzalo and Lozano-Dur{\'a}n, Adri{\'a}n},
title={Decomposing causality into its synergistic, unique, and redundant components},
journal={Nature Communications},
year={2024},
month={Nov},
day={01},
volume={15},
number={1},
pages={9296},
issn={2041-1723},
doi={10.1038/s41467-024-53373-4}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

