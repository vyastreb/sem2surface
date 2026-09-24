import numpy as np
from PIL import Image

from sem2surface_gui import _prepare_preview_image


def test_prepare_preview_image_scales_16_bit_sem_data():
    pixels = np.linspace(1024, 64512, 48, dtype=np.uint16).reshape(6, 8)
    source = Image.fromarray(pixels)

    preview = _prepare_preview_image(source)

    assert preview.mode == "RGB"
    assert preview.size == source.size
    assert preview.getextrema() == ((0, 255), (0, 255), (0, 255))


def test_prepare_preview_image_keeps_standard_rgb_values():
    pixels = np.array([[[12, 34, 56], [78, 90, 123]]], dtype=np.uint8)
    source = Image.fromarray(pixels, mode="RGB")

    preview = _prepare_preview_image(source)

    np.testing.assert_array_equal(np.asarray(preview), pixels)
