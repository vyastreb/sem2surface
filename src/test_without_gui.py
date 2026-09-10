"""Small source-tree example; installed users should prefer the CLI."""

from pathlib import Path

import sem2surface as s2s


HERE = Path(__file__).resolve().parent
images = [
    HERE / "BSE X=6, Y=0 mode A_001, HFW=500.tif",
    HERE / "BSE X=6, Y=0 mode B_001, HFW=500_002.tif",
    HERE / "BSE X=6, Y=0 mode C_001, HFW=500um_005.tif",
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
