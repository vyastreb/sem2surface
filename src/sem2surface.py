"""Core routines for reconstructing surfaces from multi-detector SEM images."""

from __future__ import annotations

import datetime as _datetime
import re
import warnings
from pathlib import Path
from typing import IO, Sequence

import matplotlib

# The core only writes figures. A non-interactive backend also makes the module
# safe to call from the GUI worker thread and on headless machines.
matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import radon


__version__ = "0.2.1"
DEFAULT_PIXEL_SIZE = 1e-6
_PIXEL_WIDTH_TAGS = ("PixelWidth=", "Image Pixel Size =")

plt.rcParams["font.family"] = "serif"


def write_vtk(filename: str | Path, X: np.ndarray, Y: np.ndarray, z: np.ndarray) -> None:
    """Write a surface as a VTK XML structured grid."""
    try:
        import vtk
    except ImportError as exc:  # pragma: no cover - optional package
        raise RuntimeError(
            "VTK export requires: pip install 'sem2surface[vtk]'"
        ) from exc

    X, Y, Z = np.asarray(X), np.asarray(Y), np.asarray(z)
    if not (X.shape == Y.shape == Z.shape):
        raise ValueError("X, Y, and Z must have the same dimensions")
    if X.ndim != 2:
        raise ValueError("X, Y, and Z must be two-dimensional")

    ny, nx = X.shape
    points = vtk.vtkPoints()
    for j in range(ny):
        for i in range(nx):
            points.InsertNextPoint(float(X[j, i]), float(Y[j, i]), float(Z[j, i]))

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nx, ny, 1)
    grid.SetPoints(points)

    z_values = vtk.vtkDoubleArray()
    z_values.SetName("Z-Value")
    z_values.SetNumberOfComponents(1)
    z_values.SetNumberOfTuples(nx * ny)
    for j in range(ny):
        for i in range(nx):
            z_values.SetValue(j * nx + i, float(Z[j, i]))
    grid.GetPointData().SetScalars(z_values)

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(str(filename))
    writer.SetInputData(grid)
    if writer.Write() != 1:
        raise OSError(f"VTK failed to write {filename}")


def log(log_file: IO[str], text: str) -> None:
    """Write a message to both stdout and the reconstruction log."""
    print("*     " + text)
    log_file.write(text + "\n")
    log_file.flush()


def parabolic_surface(params: Sequence[float], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return a centred separable paraboloid with radii ``a`` and ``b``."""
    a, b, c = params
    x0 = 0.5 * (np.max(X) + np.min(X))
    y0 = 0.5 * (np.max(Y) + np.min(Y))
    return (X - x0) ** 2 / (2 * a) + (Y - y0) ** 2 / (2 * b) + c


def objective_function(
    params: Sequence[float], X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> float:
    return float(np.sum((Z - parabolic_surface(params, X, Y)) ** 2))


def remove_outside_central_circle(img: np.ndarray) -> np.ndarray:
    """Return a copy with pixels outside the largest central circle set to zero."""
    modified = np.asarray(img).copy()
    rows, columns = np.ogrid[: modified.shape[0], : modified.shape[1]]
    radius = min(modified.shape) // 2
    center_row = modified.shape[0] // 2
    center_column = modified.shape[1] // 2
    outside = (rows - center_row) ** 2 + (columns - center_column) ** 2 > radius**2
    modified[outside] = 0
    return modified


def _parse_length(value: str) -> float | None:
    match = re.search(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-zA-Zµμ]*)",
        value,
    )
    if not match:
        return None
    magnitude = float(match.group(1))
    unit = match.group(2).lower().replace("μ", "µ")
    factors = {
        "": 1.0,
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "nm": 1e-9,
        "nanometer": 1e-9,
        "nanometers": 1e-9,
        "um": 1e-6,
        "µm": 1e-6,
        "micron": 1e-6,
        "microns": 1e-6,
        "micrometer": 1e-6,
        "micrometers": 1e-6,
        "mm": 1e-3,
        "cm": 1e-2,
    }
    factor = factors.get(unit)
    if factor is None:
        return None
    result = magnitude * factor
    return result if np.isfinite(result) and result > 0 else None


def get_pixel_width(filename: str | Path) -> float:
    """Read the physical pixel width, in metres, from SEM TIFF metadata."""
    path = Path(filename)
    content = path.read_bytes().decode("ISO-8859-1")
    for tag in _PIXEL_WIDTH_TAGS:
        start = content.find(tag)
        if start < 0:
            continue
        start += len(tag)
        ends = [
            position
            for position in (content.find("\n", start), content.find("\x00", start))
            if position >= 0
        ]
        raw_value = content[start : min(ends, default=len(content))].strip()
        value = _parse_length(raw_value)
        if value is not None:
            return value
    raise ValueError(
        f"No physical pixel width was found in {path.name}; enter it manually."
    )


def reconstruct_surface_fft(
    gradient_rows: np.ndarray, gradient_columns: np.ndarray, cutoff: float = 0.0
) -> np.ndarray:
    """Integrate two gradients using the Frankot-Chellappa FFT method.

    ``cutoff`` is a fraction of the Nyquist frequency and must be in [0, 1].
    Historical frequency normalization is retained so existing calibration
    factors remain valid, while ``fftfreq`` fixes odd-sized images.
    """
    gx = np.asarray(gradient_rows, dtype=np.float64)
    gy = np.asarray(gradient_columns, dtype=np.float64)
    if gx.shape != gy.shape or gx.ndim != 2:
        raise ValueError("Both gradients must be two-dimensional and have the same shape")
    if not 0 <= cutoff <= 1:
        raise ValueError("FFT cutoff must be between 0 and 1")

    rows, columns = gx.shape
    row_frequencies = np.fft.fftfreq(rows) * rows
    column_frequencies = np.fft.fftfreq(columns) * columns
    k_columns, k_rows = np.meshgrid(column_frequencies, row_frequencies)

    transformed_x = np.fft.fft2(gx)
    transformed_y = np.fft.fft2(gy)
    if cutoff > 0:
        cutoff_squared = (min(rows, columns) * cutoff / 2) ** 2
        high_frequency = k_rows**2 + k_columns**2 > cutoff_squared
        transformed_x[high_frequency] = 0
        transformed_y[high_frequency] = 0

    denominator = k_rows**2 + k_columns**2
    transformed_surface = np.zeros_like(transformed_x, dtype=np.complex128)
    nonzero = denominator > 0
    transformed_surface[nonzero] = (
        -1j
        * (
            k_rows[nonzero] * transformed_x[nonzero]
            + k_columns[nonzero] * transformed_y[nonzero]
        )
        / denominator[nonzero]
    )
    surface = np.fft.ifft2(transformed_surface).real
    return surface - np.mean(surface)


# Backwards-compatible spelling used by v0.1 scripts.
reconstruct_surface_FFT = reconstruct_surface_fft


def compute_image_gradients(
    imgs: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the intensity image and two normalized principal images."""
    if len(imgs) < 3:
        raise ValueError("At least three detector images are required")
    shapes = {np.asarray(img).shape for img in imgs}
    if len(shapes) != 1:
        raise ValueError("All input images must have the same shape")
    if len(next(iter(shapes))) != 2:
        raise ValueError("Detector images must be two-dimensional")

    image_stack = np.asarray(imgs, dtype=np.float64)
    image_matrix = image_stack.reshape(image_stack.shape[0], -1)
    correlation = image_matrix @ image_matrix.T
    eigenvectors, _, _ = np.linalg.svd(correlation, full_matrices=False)

    # Eigenvector signs are arbitrary. Fix each one by its largest loading so
    # different BLAS implementations produce consistent output.
    for component in range(3):
        pivot = int(np.argmax(np.abs(eigenvectors[:, component])))
        if eigenvectors[pivot, component] < 0:
            eigenvectors[:, component] *= -1

    principal = np.tensordot(eigenvectors[:, :3].T, image_stack, axes=(1, 0))
    intensity, component_1, component_2 = principal
    if np.mean(intensity) < 0:
        intensity *= -1
        component_1 *= -1
        component_2 *= -1

    threshold = max(float(np.max(np.abs(intensity))) * 1e-12, np.finfo(float).tiny)
    signs = np.where(intensity < 0, -1.0, 1.0)
    safe_intensity = np.where(np.abs(intensity) > threshold, intensity, signs * threshold)
    return intensity, component_1 / safe_intensity, component_2 / safe_intensity


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an RGB/RGBA array to floating-point grayscale."""
    array = np.asarray(img)
    if array.ndim == 2:
        return array
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    return np.dot(array[..., :3], [0.299, 0.587, 0.114])


def _read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    return np.asarray(convert_to_grayscale(array), dtype=np.float64)


def _find_sem_data_rows(image: np.ndarray) -> int:
    """Return the number of rows above a SEM annotation footer.

    FEI and Zeiss exports use different bit depths and footer layouts, but both
    place a nearly uniform bright separator near the bottom of the image.  A
    separator is accepted only when the pixels below it have a substantially
    different median intensity, which avoids treating an isolated bright line
    in the specimen as a footer.
    """
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Footer detection requires a two-dimensional image")
    rows, columns = array.shape
    if rows < 16 or columns < 16:
        return rows

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return rows
    low, high = np.percentile(finite, (0.1, 99.9))
    intensity_span = float(high - low)
    if not np.isfinite(intensity_span) or intensity_span <= 0:
        return rows

    # The 3% tolerance includes FEI separator values (64512) when a few
    # saturated specimen pixels raise the 99.9th percentile to 65535.
    bright_threshold = high - 0.03 * intensity_span
    bright_fraction = np.mean(array >= bright_threshold, axis=1)
    first_candidate = max(1, int(np.ceil(0.5 * rows)))
    minimum_footer_rows = max(4, int(np.ceil(0.01 * rows)))
    candidate_stop = rows - minimum_footer_rows + 1

    for row in range(first_candidate, candidate_stop):
        if bright_fraction[row] < 0.98 or bright_fraction[row - 1] >= 0.5:
            continue
        preceding_rows = min(64, row)
        specimen_median = float(np.nanmedian(array[row - preceding_rows : row]))
        footer_median = float(np.nanmedian(array[row:]))
        if abs(footer_median - specimen_median) >= 0.10 * intensity_span:
            return row
    return rows


def _plot_image_decomposition(
    imgs: np.ndarray,
    intensity: np.ndarray,
    gradient_1: np.ndarray,
    gradient_2: np.ndarray,
    filename: Path,
) -> None:
    columns = max(len(imgs), 3)
    fig, axes = plt.subplots(2, columns, figsize=(4 * columns, 8), squeeze=False)
    for axis in axes.flat:
        axis.set_axis_off()
    for index, image in enumerate(imgs):
        axes[0, index].imshow(image, cmap="gray")
        axes[0, index].set_title(f"Detector image {index + 1}")
    components = (
        (intensity, "Principal intensity image"),
        (gradient_1, "Normalized principal image 2"),
        (gradient_2, "Normalized principal image 3"),
    )
    for index, (image, title) in enumerate(components):
        axes[1, index].imshow(image)
        axes[1, index].set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def _plot_radon_rms(
    angles: np.ndarray,
    rms: np.ndarray,
    theta_1: float,
    theta_2: float,
    filename: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    maximum = float(np.max(rms))
    normalized = rms / maximum if maximum > 0 else np.zeros_like(rms)
    axis.plot(angles, normalized, "k-", label="Radon RMS")
    axis.scatter(angles, normalized, c="green", marker="o", s=20, zorder=10)
    axis.set_ylim(None, 1.0)
    for angle, height, label in ((theta_1, 0.5, "1"), (theta_2, 0.1, "2")):
        axis.axvline(x=angle, color="k", linestyle="--")
        axis.text(angle + 2, height, f"theta_{label} = {angle:.2f} deg", color="k")
    axis.set_xlim(0, 180)
    axis.set_xlabel("Angle (degrees)")
    axis.set_ylabel("Normalized Radon RMS value")
    axis.set_title("Radon transform RMS vs angle")
    axis.grid(True, linestyle="--", alpha=0.7)
    axis.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def _find_principal_angle(
    gradient: np.ndarray, log_file: IO[str]
) -> tuple[float, np.ndarray, np.ndarray]:
    dissection = 10
    angle_start = 0.0
    angle_end = 180.0
    all_angles: list[float] = []
    all_rms: list[float] = []
    circular_gradient = remove_outside_central_circle(gradient)

    for iteration in range(5):
        log(
            log_file,
            f"   / Radon search: iteration {iteration} "
            f"angle_start = {angle_start} angle_end = {angle_end}",
        )
        theta = np.linspace(angle_start, angle_end, dissection, endpoint=False)
        transformed = radon(circular_gradient, theta=theta, circle=True)
        rms = np.sum((transformed - np.mean(transformed, axis=0)) ** 2, axis=0)
        all_angles.extend(theta.tolist())
        all_rms.extend(rms.tolist())
        theta_1 = float(theta[np.argmin(rms)])
        # Preserve the v0.1 refinement trajectory: the published calibration
        # examples and their scaling factors were obtained with this narrowing
        # rule. Changing it would silently require recalibration.
        previous_end = angle_end
        angle_start = theta_1 - 2 * (previous_end - angle_start) / dissection
        angle_end = theta_1 + 2 * (previous_end - angle_start) / dissection

    theta_1 %= 180.0
    angles = np.mod(np.asarray(all_angles), 180.0)
    values = np.asarray(all_rms)
    order = np.argsort(angles)
    return theta_1, angles[order], values[order]


def _remove_curvature(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    mode: str,
    manual_rx: float | None,
    manual_ry: float | None,
    log_file: IO[str],
) -> tuple[np.ndarray, str]:
    if mode == "manual":
        if manual_rx is None or manual_ry is None or manual_rx == 0 or manual_ry == 0:
            raise ValueError("Manual curvature correction requires nonzero Rx and Ry")
        rx_um = manual_rx * 1e6
        ry_um = manual_ry * 1e6
        base = parabolic_surface((rx_um, ry_um, 0.0), X, Y)
        shift = float(np.mean(z - base))
        result = z - parabolic_surface((rx_um, ry_um, shift), X, Y)
        log(
            log_file,
            f"Curvature manually removed: Rx = {manual_rx:.2e} m, "
            f"Ry = {manual_ry:.2e} m, optimal dz = {shift:.2f} um",
        )
        return result, ""
    if mode != "automatic":
        raise ValueError("Curvature mode must be 'automatic' or 'manual'")

    try:
        center_x = 0.5 * (np.max(X) + np.min(X))
        center_y = 0.5 * (np.max(Y) + np.min(Y))
        scale_x = max(0.5 * (np.max(X) - np.min(X)), 1.0)
        scale_y = max(0.5 * (np.max(Y) - np.min(Y)), 1.0)
        x_squared = ((X - center_x) / scale_x) ** 2
        y_squared = ((Y - center_y) / scale_y) ** 2
        design = np.column_stack(
            (x_squared.ravel(), y_squared.ravel(), np.ones(z.size))
        )
        coefficients, _, rank, _ = np.linalg.lstsq(design, z.ravel(), rcond=None)
        if rank < 3 or not np.all(np.isfinite(coefficients)):
            raise RuntimeError("the paraboloid fit is rank-deficient")
        coefficient_x = coefficients[0] / scale_x**2
        coefficient_y = coefficients[1] / scale_y**2
        if coefficient_x == 0 or coefficient_y == 0:
            raise RuntimeError("the fitted curvature is zero")
        rx_fit = 1.0 / (2.0 * coefficient_x)
        ry_fit = 1.0 / (2.0 * coefficient_y)
        dz_fit = float(coefficients[2])
        fitted_surface = (
            coefficients[0] * x_squared
            + coefficients[1] * y_squared
            + coefficients[2]
        )
        result = z - fitted_surface
        if rx_fit * ry_fit < 0:
            message = (
                "Warning: Wrong order of images. The result is not reliable. "
                "Reshuffle images and run again."
            )
            log(log_file, f"WARNING: {message} Rx = {rx_fit:.2f} um, Ry = {ry_fit:.2f} um")
        else:
            message = ""
            log(
                log_file,
                f"Curvature removed: Rx = {rx_fit:.2f} um, Ry = {ry_fit:.2f} um, "
                f"dz = {dz_fit:.2f} um",
            )
            if rx_fit > 0 and ry_fit > 0:
                result *= -1
                log(log_file, "The reconstructed surface was flipped.")
        return result, message
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        log(log_file, f"Curvature removal skipped because fitting failed: {exc}")
        return z, f"Warning: curvature removal failed: {exc}"


def construct_surface(
    img_names: Sequence[str | Path],
    *,
    plot_images_decomposition: bool = False,
    gaussian_filter_enabled: bool = False,
    sigma: float = 1.0,
    remove_curvature: bool = False,
    curvature_mode: str = "automatic",
    manual_rx: float | None = None,
    manual_ry: float | None = None,
    cutoff_frequency: float = 0.0,
    save_file_type: str = "",
    time_stamp: bool = False,
    pixelsize: float | None = None,
    z_scaling_factor_per_pixel: float = 1.0,
    output_dir: str | Path = ".",
    log_file: IO[str] | None = None,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, str]:
    """Reconstruct a surface from three or more detector images.

    Pixel size and curvature radii are expressed in metres. Returned X, Y and Z
    arrays are expressed in micrometres.
    """
    paths = [Path(name).expanduser() for name in img_names]
    if len(paths) < 3:
        raise ValueError("At least three detector images are required")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Input image(s) not found: " + ", ".join(missing))
    if sigma < 0:
        raise ValueError("Gaussian sigma cannot be negative")
    if not 0 <= cutoff_frequency <= 1:
        raise ValueError("FFT cutoff must be between 0 and 1")
    if pixelsize is None:
        pixelsize = get_pixel_width(paths[0])
    if not np.isfinite(pixelsize) or pixelsize <= 0:
        raise ValueError("Pixel size must be a positive number in metres")
    if not np.isfinite(z_scaling_factor_per_pixel):
        raise ValueError("Z scaling factor must be finite")

    output_directory = Path(output_dir).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = (
        "_" + _datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if time_stamp
        else ""
    )
    owns_log = log_file is None
    if log_file is None:
        log_file = (output_directory / f"log{timestamp}.log").open("a", encoding="utf-8")
    try:
        return _construct_surface(
            paths,
            plot_images_decomposition=plot_images_decomposition,
            gaussian_filter_enabled=gaussian_filter_enabled,
            sigma=sigma,
            remove_curvature=remove_curvature,
            curvature_mode=curvature_mode,
            manual_rx=manual_rx,
            manual_ry=manual_ry,
            cutoff_frequency=cutoff_frequency,
            save_file_type=save_file_type,
            time_stamp=timestamp,
            pixelsize=float(pixelsize),
            z_scaling_factor_per_pixel=float(z_scaling_factor_per_pixel),
            output_directory=output_directory,
            log_file=log_file,
        )
    finally:
        if owns_log:
            log_file.close()


def _construct_surface(
    paths: Sequence[Path],
    *,
    plot_images_decomposition: bool,
    gaussian_filter_enabled: bool,
    sigma: float,
    remove_curvature: bool,
    curvature_mode: str,
    manual_rx: float | None,
    manual_ry: float | None,
    cutoff_frequency: float,
    save_file_type: str,
    time_stamp: str,
    pixelsize: float,
    z_scaling_factor_per_pixel: float,
    output_directory: Path,
    log_file: IO[str],
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, str]:
    log(log_file, f"All output is saved in {output_directory.resolve()}")
    log(log_file, "Parameters:")
    log(log_file, f"   / Detector images = {len(paths)}")
    log(log_file, f"   / Plot intermediate images = {plot_images_decomposition}")
    log(log_file, f"   / Gaussian filter = {gaussian_filter_enabled}")
    log(log_file, f"   / Gaussian sigma = {sigma}")
    log(log_file, f"   / Remove curvature = {remove_curvature}")
    log(log_file, f"   / FFT cutoff frequency = {cutoff_frequency}")
    log(log_file, f"   / Output file type = {save_file_type or 'do not save'}")
    log(log_file, f"   / Pixel size = {pixelsize} m")
    log(log_file, f"   / Z scaling factor per pixel = {z_scaling_factor_per_pixel} 1/m")
    log(log_file, f"Images folder: {paths[0].resolve().parent}")
    log(log_file, "Image names:")
    for path in paths:
        log(log_file, f"    / {path.name}")

    images = [_read_image(path) for path in paths]
    widths = {image.shape[1] for image in images}
    if len(widths) != 1:
        raise ValueError("All detector images must have the same width")

    # Detect each footer independently, then apply the smallest common crop so
    # all detector images retain identical dimensions.
    crop_rows = [_find_sem_data_rows(image) for image in images]
    for path, image, crop in zip(paths, images, crop_rows):
        if crop < image.shape[0]:
            log(
                log_file,
                f"SEM annotation footer detected in {path.name}: "
                f"removed {image.shape[0] - crop} rows",
            )
    cut_y = min(crop_rows)
    if cut_y < 2:
        raise ValueError("Automatic footer detection left too little image data")
    log(log_file, f"SEM data rows retained: {cut_y}")
    image_stack = np.stack(
        [np.nan_to_num(image[:cut_y, :], copy=False) for image in images], axis=0
    )

    if gaussian_filter_enabled and sigma > 0:
        image_stack = gaussian_filter(image_stack, sigma=(0, sigma, sigma))
        log(log_file, f"Gaussian filter with sigma = {sigma} applied to all images.")

    intensity, gradient_1, gradient_2 = compute_image_gradients(image_stack)
    if plot_images_decomposition:
        filename = output_directory / f"Images_decomposition{time_stamp}.png"
        _plot_image_decomposition(image_stack, intensity, gradient_1, gradient_2, filename)
        log(log_file, f"Images decomposition saved to {filename}")

    theta_1, angles, rms = _find_principal_angle(gradient_1, log_file)
    theta_2 = (theta_1 + 90.0) % 180.0
    log(log_file, f"theta1 = {theta_1}")
    log(log_file, f"theta2 = {theta_2}")
    if plot_images_decomposition:
        filename = output_directory / f"RadonTransformRMS{time_stamp}.pdf"
        _plot_radon_rms(angles, rms, theta_1, theta_2, filename)
        log(log_file, f"Radon transform RMS saved to {filename}")

    radians_1, radians_2 = np.deg2rad((theta_1, theta_2))
    gradient_rows = np.cos(radians_1) * gradient_1 + np.cos(radians_2) * gradient_2
    gradient_columns = np.sin(radians_1) * gradient_1 + np.sin(radians_2) * gradient_2
    gradient_rows -= np.mean(gradient_rows)
    gradient_columns -= np.mean(gradient_columns)

    if plot_images_decomposition:
        figure = plt.figure(figsize=(15, 8))
        grid = gridspec.GridSpec(2, 2, height_ratios=[1, 0.05])
        for index, (gradient, title) in enumerate(
            ((gradient_rows, "Gradient along rows"), (gradient_columns, "Gradient along columns"))
        ):
            axis = figure.add_subplot(grid[0, index])
            color_axis = figure.add_subplot(grid[1, index])
            shown = axis.imshow(gradient)
            figure.colorbar(shown, cax=color_axis, orientation="horizontal")
            axis.set_title(title)
        figure.tight_layout()
        filename = output_directory / f"Gradients{time_stamp}.png"
        figure.savefig(filename, dpi=300)
        plt.close(figure)
        log(log_file, f"Gradient images saved to {filename}")

    z = reconstruct_surface_fft(gradient_rows, gradient_columns, cutoff_frequency)
    z *= 1e6 * z_scaling_factor_per_pixel * pixelsize

    rows, columns = z.shape
    pre_x, pre_y = np.meshgrid(
        np.arange(columns) * pixelsize * 1e6,
        np.arange(rows) * pixelsize * 1e6,
    )
    return_message = ""
    if remove_curvature:
        z, return_message = _remove_curvature(
            z, pre_x, pre_y, curvature_mode, manual_rx, manual_ry, log_file
        )

    z = np.rot90(z)
    x = np.arange(z.shape[1]) * pixelsize * 1e6
    y = np.arange(z.shape[0]) * pixelsize * 1e6
    X, Y = np.meshgrid(x, y)

    figure, axis = plt.subplots(figsize=(8, 10))
    extent = [x[-1] if x.size else 0, 0, 0, y[-1] if y.size else 0]
    shown = axis.imshow(z, extent=extent, interpolation="none")
    axis.set_xlabel("y (micrometres)")
    axis.set_ylabel("x (micrometres)")
    axis.set_title("Reconstructed surface")
    divider = make_axes_locatable(axis)
    color_axis = divider.append_axes("top", size="5%", pad=0.5)
    colorbar = figure.colorbar(shown, cax=color_axis, orientation="horizontal")
    colorbar.set_label("z (micrometres)")
    color_axis.xaxis.set_ticks_position("top")
    color_axis.xaxis.set_label_position("top")
    figure.tight_layout()
    surface_image = output_directory / f"Surface_FFT{time_stamp}.png"
    figure.savefig(surface_image, dpi=300, bbox_inches="tight")
    plt.close(figure)
    log(log_file, f"FFT-reconstructed surface saved to {surface_image}")

    figure, axis = plt.subplots(figsize=(8, 8 * z.shape[0] / z.shape[1]))
    axis.imshow(z, cmap="gray")
    axis.set_axis_off()
    axis.set_aspect("auto")
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    grayscale_image = output_directory / f"Surface_BW_FFT{time_stamp}.png"
    figure.savefig(grayscale_image, dpi=300)
    plt.close(figure)

    output_type = save_file_type.strip().upper()
    if output_type == "CSV":
        filename = output_directory / f"Surface{time_stamp}.csv"
        with filename.open("w", encoding="utf-8", newline="") as stream:
            stream.write("# x (um), y (um), z (um)\n")
            for row in range(z.shape[0]):
                for column in range(z.shape[1]):
                    stream.write(
                        f"{X[row, column]:.6f},{Y[row, column]:.6f},"
                        f"{z[row, column]:.6f}\n"
                    )
        log(log_file, f"Surface saved to {filename}")
    elif output_type == "NPZ":
        filename = output_directory / f"Surface{time_stamp}.npz"
        np.savez(filename, X=X, Y=Y, Z=z)
        log(log_file, f"Surface saved to {filename}")
    elif output_type == "VTK":
        filename = output_directory / f"Surface{time_stamp}.vts"
        write_vtk(filename, X, Y, z)
        log(log_file, f"Surface saved to {filename}")
    elif output_type not in ("", "DO NOT SAVE", "NONE"):
        raise ValueError("Output type must be CSV, NPZ, VTK, or empty")
    else:
        log(log_file, "Surface data were not saved")

    log(log_file, f"RMS of the surface = {np.std(z)}")
    log(
        log_file,
        "Successfully finished at " + _datetime.datetime.now().isoformat(timespec="seconds"),
    )
    return str(surface_image), X, Y, z, return_message


def constructSurface(
    imgNames: Sequence[str | Path],
    Plot_images_decomposition: bool = False,
    GaussFilter: bool = False,
    sigma: float = 1.0,
    ReconstructionMode: str = "FFT",
    RemoveCurvature: bool = False,
    curvature_mode: str = "automatic",
    manual_rx: float | None = None,
    manual_ry: float | None = None,
    cutoff_frequency: float = 0.0,
    save_file_type: str = "",
    time_stamp: bool = False,
    pixelsize: float | None = None,
    ZscalingFactorPerPixel: float = 1.0,
    Z_ref: float | None = None,
    Z_current: float | None = None,
    logFile: IO[str] | None = None,
    output_dir: str | Path = ".",
):
    """Compatibility wrapper for the v0.1 camelCase API."""
    if ReconstructionMode != "FFT":
        raise ValueError("Direct integration was removed; only FFT reconstruction is supported")
    if Z_ref is not None or Z_current is not None:
        warnings.warn(
            "Z_ref and Z_current are deprecated and ignored; use the calibrated scaling factor only",
            DeprecationWarning,
            stacklevel=2,
        )
    return construct_surface(
        imgNames,
        plot_images_decomposition=Plot_images_decomposition,
        gaussian_filter_enabled=GaussFilter,
        sigma=sigma,
        remove_curvature=RemoveCurvature,
        curvature_mode=curvature_mode,
        manual_rx=manual_rx,
        manual_ry=manual_ry,
        cutoff_frequency=cutoff_frequency,
        save_file_type=save_file_type,
        time_stamp=time_stamp,
        pixelsize=pixelsize,
        z_scaling_factor_per_pixel=ZscalingFactorPerPixel,
        output_dir=output_dir,
        log_file=logFile,
    )
