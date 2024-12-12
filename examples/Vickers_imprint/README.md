# Reconstruction of the surface from SEM images
## Example 3: reconstruction of the Vickers imprint

## Usage

```bash
python test_without_gui.py
```
To show the reconstruction in VTK, you can use the following command:
```bash
paraview Surface.vts
```

## Description

This script makes use of the sem2surface.py script to reconstruct the surface from SEM images.

Script that runs the reconstruction:

- `test_without_gui.py` - it contains the main parameters for the reconstruction and a list of input files.

Input files (SEM images from 3 detectors):

- `Indent_20230912_7kV_spot4.5_FOV100um_multiBSE_C_05.tif`
- `Indent_20230912_7kV_spot4.5_FOV100um_multiBSE_C_05.tif`
- `Indent_20230912_7kV_spot4.5_FOV100um_multiBSE_C_05.tif`

Output file including extra files produced by the sem2surface.py script:

- `Images_decomposition.png` - Decomposition of the SEM images into 3 components
- `Gradients.png` - Reconstructed normalized gradients along $x$ and $y$ axes
- `log.log` - Log file with the reconstruction process and all the parameters used
- `RadonTransformRMS.pdf` - PDF file with the RMS of the Radon transform
- `Surface_FFT.png` - Image of the reconstructed surface using FFT
- `Surface_BW_FFT.png` - Black and white image of the reconstructed surface
- `Surface.vts` - VTK file with the reconstructed surface
- `VTK_view.png` - Image showing a zoom of the reconstructed surface using VTK

