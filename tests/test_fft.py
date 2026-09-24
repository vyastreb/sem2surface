import numpy as np
import pytest

from sem2surface import reconstruct_surface_fft


@pytest.mark.parametrize("size", [31, 32, 33, 64])
def test_fft_reconstructs_even_and_odd_synthetic_surfaces(size):
    row, column = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    frequencies = (1, 2, 5, 9)
    surface = sum(
        np.sin(2 * np.pi * frequency * row / size + 0.2 * frequency)
        + 0.6 * np.cos(2 * np.pi * frequency * column / size - 0.1 * frequency)
        for frequency in frequencies
    )
    gradient_rows = sum(
        (2 * np.pi * frequency / size)
        * np.cos(2 * np.pi * frequency * row / size + 0.2 * frequency)
        for frequency in frequencies
    )
    gradient_columns = sum(
        -0.6
        * (2 * np.pi * frequency / size)
        * np.sin(2 * np.pi * frequency * column / size - 0.1 * frequency)
        for frequency in frequencies
    )

    reconstructed = reconstruct_surface_fft(gradient_rows, gradient_columns)

    # Historical sem2surface normalization is 2*pi/N for square images.
    np.testing.assert_allclose(reconstructed * size / (2 * np.pi), surface, atol=1e-12)


def test_fft_cutoff_removes_high_frequency_content():
    size = 64
    row, column = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    low = np.sin(2 * np.pi * 2 * row / size)
    high = 0.5 * np.sin(2 * np.pi * 20 * row / size)
    gradient_rows = (
        (2 * np.pi * 2 / size) * np.cos(2 * np.pi * 2 * row / size)
        + 0.5 * (2 * np.pi * 20 / size) * np.cos(2 * np.pi * 20 * row / size)
    )
    gradient_columns = np.zeros_like(column, dtype=float)

    reconstructed = reconstruct_surface_fft(gradient_rows, gradient_columns, cutoff=0.25)

    np.testing.assert_allclose(reconstructed * size / (2 * np.pi), low, atol=1e-12)
    assert not np.allclose(reconstructed * size / (2 * np.pi), low + high)


@pytest.mark.parametrize("cutoff", [-0.1, 1.1])
def test_fft_rejects_invalid_cutoff(cutoff):
    gradient = np.ones((8, 8))
    with pytest.raises(ValueError, match="between 0 and 1"):
        reconstruct_surface_fft(gradient, gradient, cutoff)
