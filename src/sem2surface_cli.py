"""Command-line interface for sem2surface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sem2surface import __version__ as VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sem2surface",
        description="Reconstruct a 3D surface from three or more SEM/BSE detector images.",
    )
    parser.add_argument("images", nargs="+", type=Path, help="detector images in acquisition order")
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        help="pixel size in micrometres; by default it is read from the first TIFF",
    )
    parser.add_argument(
        "--z-scale",
        type=float,
        default=2.1727243e2,
        help="calibrated Z scaling factor per pixel in 1/m (default: %(default)g)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.0,
        metavar="FRACTION",
        help="FFT cutoff as a fraction of Nyquist, from 0 to 1",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        metavar="PIXELS",
        help="enable Gaussian filtering with this standard deviation",
    )
    parser.add_argument(
        "--curvature",
        choices=("none", "automatic", "manual"),
        default="automatic",
        help="curvature correction mode (default: %(default)s)",
    )
    parser.add_argument("--rx", type=float, help="manual X curvature radius in metres")
    parser.add_argument("--ry", type=float, help="manual Y curvature radius in metres")
    parser.add_argument(
        "--save",
        choices=("none", "csv", "npz", "vtk"),
        default="npz",
        help="surface data format (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path.cwd(), help="output directory"
    )
    parser.add_argument("--timestamp", action="store_true", help="timestamp output filenames")
    parser.add_argument(
        "--extra-images", action="store_true", help="save decomposition and gradient figures"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.images) < 3:
        parser.error("at least three detector images are required")
    if args.pixel_size_um is not None and args.pixel_size_um <= 0:
        parser.error("--pixel-size-um must be positive")
    if args.curvature == "manual" and (args.rx is None or args.ry is None):
        parser.error("manual curvature correction requires both --rx and --ry")

    from sem2surface import construct_surface

    pixelsize = args.pixel_size_um * 1e-6 if args.pixel_size_um is not None else None
    image_name, _, _, _, warning = construct_surface(
        args.images,
        plot_images_decomposition=args.extra_images,
        gaussian_filter_enabled=args.gaussian_sigma is not None,
        sigma=args.gaussian_sigma or 0.0,
        remove_curvature=args.curvature != "none",
        curvature_mode=("automatic" if args.curvature == "none" else args.curvature),
        manual_rx=args.rx,
        manual_ry=args.ry,
        cutoff_frequency=args.cutoff,
        save_file_type="" if args.save == "none" else args.save,
        time_stamp=args.timestamp,
        pixelsize=pixelsize,
        z_scaling_factor_per_pixel=args.z_scale,
        output_dir=args.output_dir,
    )
    print(f"Reconstruction image: {image_name}")
    if warning:
        print(warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
