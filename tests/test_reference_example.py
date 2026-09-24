from pathlib import Path

import numpy as np
import pytest

from sem2surface import construct_surface, get_pixel_width


REFERENCE = Path(__file__).parents[1] / "examples" / "Surface_1"


@pytest.mark.skipif(
    not (REFERENCE / "P33_scale1_detectorA.tif").is_file(),
    reason="reference images are not included in the source distribution",
)
def test_surface_1_reference_reconstruction(tmp_path):
    images = [
        REFERENCE / "P33_scale1_detectorA.tif",
        REFERENCE / "P33_scale1_detectorB.tif",
        REFERENCE / "P33_scale1_detectorC.tif",
    ]
    preview, X, Y, Z, warning = construct_surface(
        images,
        remove_curvature=True,
        pixelsize=get_pixel_width(images[0]),
        z_scaling_factor_per_pixel=2.1727243e2,
        save_file_type="NPZ",
        output_dir=tmp_path,
    )

    assert warning == ""
    assert Path(preview).is_file()
    assert (tmp_path / "Surface.npz").is_file()
    assert X.shape == Y.shape == Z.shape == (1536, 1024)
    assert np.all(np.isfinite(Z))
    assert X[0, -1] == pytest.approx(1023 * 0.325521)
    assert Y[-1, 0] == pytest.approx(1535 * 0.325521)
    assert np.std(Z) == pytest.approx(0.8193, rel=0.02)
