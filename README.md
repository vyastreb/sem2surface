# sem2surface

[![PyPI version](https://img.shields.io/pypi/v/sem2surface.svg)](https://pypi.org/project/sem2surface/)
[![License: BSD 3-Clause](https://img.shields.io/pypi/l/sem2surface.svg)](https://github.com/vyastreb/sem2surface/blob/master/LICENSE)

`sem2surface` reconstructs a three-dimensional surface from three to five
multi-detector SEM/BSE images. It extracts two normalized principal images,
identifies their orientation with a Radon transform, and integrates the resulting
gradients using the Frankot-Chellappa FFT method.

<!-- PyPI cannot resolve repository-relative images. Keep this absolute URL. -->
![3D surface reconstruction from multi-detector SEM images](https://raw.githubusercontent.com/vyastreb/sem2surface/master/img/explication.jpg)

The reconstruction can be used qualitatively with an arbitrary vertical scale.
Quantitative measurements require a calibrated Z scaling factor for the imaging
configuration. The included Vickers-indentation example demonstrates that
calibration procedure.

## Installation

Python 3.10 or newer is required. Install `sem2surface` from PyPI with:

```bash
pip install sem2surface
```

If several Python installations are present, use the interpreter explicitly:

```bash
python3 -m pip install sem2surface  # Linux or macOS
```

### Windows

Install Python 3.10 or newer from
[python.org](https://www.python.org/downloads/windows/). Keep the standard
`pip`, Tcl/Tk, and Python Launcher components enabled. Then open PowerShell or
Command Prompt and run:

```powershell
py -m pip install sem2surface
sem2surface-gui
```

If Windows cannot find the installed launcher, start the GUI through Python:

```powershell
py -m sem2surface_gui
```

On Linux, Tkinter may be packaged separately. For example, Ubuntu and Debian
users can install `python3-tk` with their system package manager.

A virtual environment is optional. It is useful when the system Python is
externally managed, installation permissions are restricted, or other packages
have conflicting dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install sem2surface
```

VTK export is optional because VTK is a large dependency:

```bash
python -m pip install "sem2surface[vtk]"
# From this source checkout instead: python -m pip install ".[vtk]"
```

The application does not require ImageMagick or a LaTeX installation. Tkinter is
included with the standard Python installers on Windows and macOS.

## Desktop application

After installation, launch the graphical interface from a terminal:

```bash
sem2surface-gui
```

Select three to five detector images, confirm the pixel size and scaling factor,
choose an output directory, and start the reconstruction. TIFF, PNG, JPEG, and
BMP images are supported. The work runs in the background so the window remains
responsive.

The GUI intentionally provides only the FFT reconstruction. Atomic-number
correction and direct profile integration have been removed.

## Command line

Installation also provides a `sem2surface` executable:

```bash
sem2surface detector_A.tif detector_B.tif detector_C.tif \
  --z-scale 217.27243 \
  --save npz \
  --output-dir results
```

Pixel size is read from the first TIFF by default. For images without compatible
SEM metadata, provide it in micrometres:

```bash
sem2surface detector_A.png detector_B.png detector_C.png \
  --pixel-size-um 0.325521 \
  --curvature none
```

Run `sem2surface --help` for the complete list of options. VTK output requires
the optional `vtk` installation described above.

## Python API

```python
from pathlib import Path

from sem2surface import construct_surface

images = [Path("detector_A.tif"), Path("detector_B.tif"), Path("detector_C.tif")]
preview, X, Y, Z, warning = construct_surface(
    images,
    z_scaling_factor_per_pixel=217.27243,
    remove_curvature=True,
    save_file_type="NPZ",
    output_dir="results",
)
```

Pixel size and manual curvature radii use metres in the Python API. Returned
coordinate and height arrays use micrometres. The old `constructSurface`
function remains as a compatibility wrapper for version 0.1 scripts, but only
FFT reconstruction is accepted and the old atomic-number arguments are ignored.

## Outputs

Every reconstruction creates:

- a colour PNG of the reconstructed surface;
- a grayscale PNG suitable for further image analysis;
- a UTF-8 log containing parameters, input names, angles, and surface RMS.

Surface arrays can additionally be saved as CSV, compressed NumPy NPZ, or VTK
structured-grid data. Optional diagnostic output includes the detector/PCA
decomposition, Radon search, and oriented gradients.

When timestamps are disabled, an existing output with the same name is replaced.
Choose a dedicated output folder or enable timestamps when results must be kept.

For TIFF acquisitions, `sem2surface` automatically removes a microscope
annotation footer when it detects a nearly uniform bright separator followed by
a statistically distinct lower band. The detector works with both 8-bit Zeiss
and 16-bit FEI exports, leaves footer-free images unchanged, and records every
detected crop in the processing log.

## Reference examples and scaling

The [examples directory](https://github.com/vyastreb/sem2surface/tree/master/examples)
contains the reference analyses:

- `Surface_1` and `Surface_2`: representative reconstructed surfaces;
- `Vickers_imprint`: reconstruction of a Vickers indentation;
- `Vickers_imprint_scaling`: identification of the Z scaling factor from the
  known Vickers geometry.

After installing the package with the VTK extra, an example can be run from any
working directory:

```bash
python examples/Surface_1/test_without_gui.py
```

<!-- PyPI cannot resolve repository-relative images. Keep this absolute URL. -->
![Reconstruction of the indented surface](https://raw.githubusercontent.com/vyastreb/sem2surface/master/img/indent_superposition.jpg)

## Changes in 0.2.2

- Replaced the historical hard-coded footer threshold with bit-depth-independent
  SEM annotation-band detection.
- Added safeguards against cropping isolated bright lines in specimen data.
- Added per-image logging of detected footers and removed row counts.
- Added regression coverage for 8-bit Zeiss, 16-bit FEI, footer-free, and
  isolated-line images.

## Development

Create an isolated environment and install the editable project with its test
tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
python -m build
python -m twine check dist/*
```

Continuous integration tests Python 3.10 and 3.13 on Linux, Windows, and macOS,
and verifies both the source distribution and universal wheel.

### Maintainer release checklist

Use a clean virtual environment and update the version in both
`pyproject.toml` and `src/sem2surface.py`. PyPI does not allow an uploaded file
or release version to be replaced.

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
```

For a first trial, create a separate account and API token on
[TestPyPI](https://test.pypi.org/), then upload only the files for the new
version:

```bash
python -m twine upload --repository testpypi dist/sem2surface-0.2.2*
```

When prompted, use `__token__` as the username and the complete TestPyPI token,
including its `pypi-` prefix, as the password. Test the uploaded wheel in a new
environment without resolving dependencies from TestPyPI:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps sem2surface==0.2.2
sem2surface --version
```

For the real release, create a PyPI account and API token at
[pypi.org](https://pypi.org/), then run:

```bash
python -m twine upload dist/sem2surface-0.2.2*
```

PyPI and TestPyPI use separate accounts and tokens. Never commit a token or put
one directly in a command. For later automated releases, prefer
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/).

## Method references

1. Neggers, J. et al. (2021). Principal image decomposition for multi-detector
   backscatter electron topography reconstruction. *Ultramicroscopy*, 227,
   113200. [DOI](https://doi.org/10.1016/j.ultramic.2020.113200)
2. Frankot, R. T. and Chellappa, R. (1988). A method for enforcing integrability
   in shape from shading algorithms. *IEEE Transactions on Pattern Analysis and
   Machine Intelligence*, 10(4), 439-451.
   [DOI](https://doi.org/10.1109/34.3909)

## Author and license

Developed by Vladislav A. Yastrebov, CNRS, Mines Paris – PSL, Centre des
matériaux. Distributed under the
[BSD 3-Clause License](https://github.com/vyastreb/sem2surface/blob/master/LICENSE).
