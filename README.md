# SEM/BSE 3D Surface Reconstruction

## Overview

This repository contains a Python-based solution for 3D surface reconstruction from SEM/BSE images captured using a minimum of three detectors. The methodology leverages the Principal Component Analysis (PCA) of the captured images to discern the principal component images [1]. To reorient gradient images along $x$ and $y$ axes, the Radon transform is used. The final 3D surface, represented as \(z(x,y)\), is derived from its gradients either through the Frankot and Chellappa method [2] or via direct integration paired with a minimization process between adjacent profiles.

## Features

- **Intuitive GUI**: A user-friendly simplistic interface built with Python's Tkinter allows for easy image uploads and 3D surface construction.
- **Comprehensive Outputs**: Each execution yields:
  - 3 original images and their PCA decomposition (PNG format).
  - Radon transform RMS change wrt the rotation angle (PDF format).
  - Gradients visualisation along $x$ and $y$ (PNG format).
  - 3D map of the reconstructed surface (2 PNG files).
  - ASCII representation of the reconstructed surface with \(x,y,z\) columns.
  - Surface roughness data (NPZ format).
  - Detailed log file capturing all operations (TXT format).

## Getting Started

To launch the interface, execute the following command:
```
$ python sem2surface.py
```
Users can easily upload a minimum of three images (supported formats: JPG, PNG, TIFF, BMP) and initiate the reconstruction process by clicking the "3D Reconstruct" button.

## Repository Structure

- `src/`
  - `sem2surface.py`: Core module for 3D surface reconstruction from SEM/BSE images.
  - `sem2surface_gui.py`: GUI module.
  - `logo.png`, `logo.svg`: Application logo.
- `doc/`
  - `SEM2surface.pdf`: Concise documentation.
- `example/`
  - Sample SEM images from different detectors.
  - Sample output of the reconstructed surface.
- `README.md`: This file.

## References

+ [1] Neggers, J., Héripré, E., Bonnet, M., Boivin, D., Tanguy, A., Hallais, S., Gaslain, F., Rouesne, E. and Roux, S. (2021). Principal image decomposition for multi-detector backscatter electron topography reconstruction. *Ultramicroscopy*, 227:113200. [DOI](https://doi.org/10.1016/j.ultramic.2020.113200)
+ [2] Frankot, R. T., & Chellappa, R. (1988). A method for enforcing integrability in shape from shading algorithms. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 10(4):439-451. [DOI](https://doi.org/10.1109/34.3909)

## Additional Information

- **Developer**: Vladislav A. Yastrebov
  - Affiliation: CNRS, MINES Paris, PSL, Centre des matériaux
  - Date: August 2023
  - [yastrebov.fr](https://yastrebov.fr)
- **Licence**: BSD 3-Clause License.

## Acknowledgements

The code was developed with the assistance of GPT-4, CoderPad plugin, and Copilot in VSCode.

