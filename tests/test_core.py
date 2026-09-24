from io import StringIO

import numpy as np
import pytest

from sem2surface import (
    _find_sem_data_rows,
    compute_image_gradients,
    construct_surface,
    convert_to_grayscale,
    get_pixel_width,
)


def test_compute_image_gradients_uses_all_five_images():
    row, column = np.meshgrid(np.arange(8), np.arange(9), indexing="ij")
    images = [100 + index * row + (5 - index) * column for index in range(5)]
    intensity, gradient_1, gradient_2 = compute_image_gradients(images)
    assert intensity.shape == (8, 9)
    assert gradient_1.shape == (8, 9)
    assert gradient_2.shape == (8, 9)
    assert np.all(np.isfinite(gradient_1))
    assert np.all(np.isfinite(gradient_2))


def test_compute_image_gradients_requires_three_images():
    with pytest.raises(ValueError, match="At least three"):
        compute_image_gradients([np.ones((4, 4)), np.ones((4, 4))])


def test_grayscale_ignores_alpha_channel():
    image = np.array([[[10, 20, 30, 0], [30, 20, 10, 255]]])
    expected = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    np.testing.assert_allclose(convert_to_grayscale(image), expected)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (b"PixelWidth=3.25521e-007\n", 3.25521e-7),
        ("Image Pixel Size = 325.521 nm\n".encode(), 325.521e-9),
    ],
)
def test_get_pixel_width(metadata, expected, tmp_path):
    image = tmp_path / "metadata.tif"
    image.write_bytes(b"TIFF" + metadata + b"payload")
    assert get_pixel_width(image) == pytest.approx(expected)


def test_construct_surface_validates_before_opening_log(tmp_path):
    with pytest.raises(ValueError, match="At least three"):
        construct_surface([], output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_caller_owned_log_is_not_closed():
    # Validation fails after accepting the caller's object, which must remain usable.
    stream = StringIO()
    with pytest.raises(ValueError):
        construct_surface([], log_file=stream)
    assert not stream.closed


def test_footer_detection_handles_fei_16_bit_separator():
    rng = np.random.default_rng(3)
    image = rng.integers(28000, 48000, size=(80, 64), dtype=np.uint16)
    image.flat[:12] = 65535  # Saturated specimen pixels above the separator.
    image[64] = 64512
    image[65:] = 1024
    image[68:76:3, 5:59:7] = 64512
    image[-1] = 64512

    assert _find_sem_data_rows(image) == 64


def test_footer_detection_handles_zeiss_8_bit_band():
    rng = np.random.default_rng(5)
    image = rng.integers(90, 190, size=(120, 64), dtype=np.uint8)
    image[101] = 0
    image[102:] = 255
    image[105:117:3, 4:60:6] = 0

    assert _find_sem_data_rows(image) == 102


def test_footer_detection_leaves_footer_free_image_unchanged():
    rng = np.random.default_rng(7)
    image = rng.integers(40, 220, size=(120, 64), dtype=np.uint8)

    assert _find_sem_data_rows(image) == image.shape[0]


def test_footer_detection_ignores_isolated_bright_specimen_line():
    rng = np.random.default_rng(11)
    image = rng.integers(80, 180, size=(120, 64), dtype=np.uint8)
    image[75] = 255

    assert _find_sem_data_rows(image) == image.shape[0]
