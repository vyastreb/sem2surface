"""Reconstruct the second reference surface."""

from pathlib import Path

import sem2surface as s2s


HERE = Path(__file__).resolve().parent
images = [
    HERE / "P33_scale2s3_detectorA.tif",
    HERE / "P33_scale2s3_detectorB.tif",
    HERE / "P33_scale2s3_detectorC.tif",
]

image_name, X, Y, Z, message = s2s.construct_surface(
    images,
    plot_images_decomposition=True,
    remove_curvature=True,
    save_file_type="VTK",
    pixelsize=s2s.get_pixel_width(images[0]),
    z_scaling_factor_per_pixel=2.1727243e2,
    output_dir=HERE,
)
print(message or f"Successfully reconstructed the surface: {image_name}")
