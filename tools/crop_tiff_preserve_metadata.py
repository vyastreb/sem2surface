#!/usr/bin/env python3
"""Crop single-page TIFF images while preserving microscope metadata.

The pixel crop is performed with Pillow.  TIFF metadata is copied into the
new IFD, nested EXIF/GPS IFDs are rebuilt, and raw BYTE/ASCII/UNDEFINED
payloads are restored byte-for-byte.  The output is verified before it is
atomically moved into place.
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image, TiffImagePlugin


# Tags describing the encoded pixel array must be regenerated after cropping.
STRUCTURAL_TAGS = {
    256,  # ImageWidth
    257,  # ImageLength
    258,  # BitsPerSample
    259,  # Compression
    262,  # PhotometricInterpretation
    266,  # FillOrder
    273,  # StripOffsets
    277,  # SamplesPerPixel
    278,  # RowsPerStrip
    279,  # StripByteCounts
    284,  # PlanarConfiguration
    317,  # Predictor
    320,  # ColorMap
    322,  # TileWidth
    323,  # TileLength
    324,  # TileOffsets
    325,  # TileByteCounts
    330,  # SubIFDs (image pyramids are intentionally unsupported)
    338,  # ExtraSamples
    339,  # SampleFormat
    347,  # JPEGTables
    513,  # JPEGInterchangeFormat
    514,  # JPEGInterchangeFormatLength
    32997,  # ImageDepth
    32998,  # TileDepth
}

# Pillow can rebuild these pointer IFDs from their dictionaries.
POINTER_IFDS = {34665, 34853, 40965}  # ExifIFD, GPSIFD, InteroperabilityIFD

# These tag types can contain vendor bytes which Pillow may decode/re-encode.
RAW_TYPES = {1, 2, 7}  # BYTE, ASCII, UNDEFINED
TYPE_SIZES = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
    13: 4,
    16: 8,
    17: 8,
    18: 8,
}


class CropError(RuntimeError):
    """Raised when safe metadata-preserving cropping is not possible."""


def _raw_payload(path: Path, tag: tifffile.TiffTag) -> bytes:
    size = TYPE_SIZES[int(tag.dtype)] * tag.count
    with path.open("rb") as stream:
        stream.seek(tag.valueoffset)
        payload = stream.read(size)
    if len(payload) != size:
        raise CropError(f"could not read TIFF tag {tag.code} from {path}")
    return payload


def _metadata_snapshot(path: Path) -> dict[int, tuple[int, int, bytes]]:
    """Return raw payloads for all non-structural, non-pointer tags."""
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) != 1:
            raise CropError(f"only single-page TIFFs are supported: {path}")
        page = tif.pages[0]
        if 330 in page.tags:
            raise CropError(f"TIFF SubIFDs/image pyramids are not supported: {path}")
        return {
            code: (int(tag.dtype), tag.count, _raw_payload(path, tag))
            for code, tag in page.tags.items()
            if code not in STRUCTURAL_TAGS and code not in POINTER_IFDS
        }


def _pointer_snapshot(path: Path) -> dict[int, Any]:
    with Image.open(path) as image:
        exif = image.getexif()
        result: dict[int, Any] = {}
        for code in POINTER_IFDS:
            if code in image.tag_v2:
                value = exif.get_ifd(code)
                if not value:
                    raise CropError(f"could not decode pointer IFD tag {code} in {path}")
                result[code] = dict(value)
        return result


def _patch_tag_count(path: Path, tag_offset: int, count: int, *, bigtiff: bool, byteorder: str) -> None:
    prefix = "<" if byteorder == "<" else ">"
    fmt = prefix + ("Q" if bigtiff else "I")
    with path.open("r+b") as stream:
        stream.seek(tag_offset + 4)
        stream.write(struct.pack(fmt, count))


def _restore_raw_metadata(
    output: Path, original: dict[int, tuple[int, int, bytes]]
) -> None:
    """Undo lossy text conversion of vendor tags, then restore their counts."""
    patches: list[tuple[int, int]] = []
    with tifffile.TiffFile(output) as tif:
        page = tif.pages[0]
        bigtiff = tif.is_bigtiff
        byteorder = tif.byteorder
        for code, (dtype, count, payload) in original.items():
            if dtype not in RAW_TYPES:
                continue
            tag = page.tags.get(code)
            if tag is None:
                raise CropError(f"output is missing TIFF metadata tag {code}")
            capacity = TYPE_SIZES[int(tag.dtype)] * tag.count
            if int(tag.dtype) != dtype or capacity < len(payload):
                raise CropError(f"cannot safely restore TIFF metadata tag {code}")
            with output.open("r+b") as stream:
                stream.seek(tag.valueoffset)
                stream.write(payload)
            if tag.count != count:
                # TIFF writers commonly append a NUL to malformed vendor ASCII.
                if dtype != 2 or tag.count != count + 1:
                    raise CropError(f"unexpected output count for TIFF tag {code}")
                patches.append((tag.offset, count))
    for offset, count in patches:
        _patch_tag_count(
            output, offset, count, bigtiff=bigtiff, byteorder=byteorder
        )


def _equal_metadata_value(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left == right
    try:
        comparison = left == right
        return bool(comparison.all()) if hasattr(comparison, "all") else bool(comparison)
    except Exception:
        return repr(left) == repr(right)


def _verify(
    source: Path,
    output: Path,
    box: tuple[int, int, int, int],
    metadata: dict[int, tuple[int, int, bytes]],
    pointers: dict[int, Any],
) -> None:
    left, top, right, bottom = box
    with tifffile.TiffFile(source) as src_tif, tifffile.TiffFile(output) as out_tif:
        source_page = src_tif.pages[0]
        output_page = out_tif.pages[0]
        expected = source_page.asarray()[top:bottom, left:right, ...]
        actual = output_page.asarray()
        if expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise CropError("output pixel dtype or shape does not match the requested crop")
        if not np.array_equal(expected, actual):
            raise CropError("output pixels differ from the requested source crop")
        for code, (dtype, count, payload) in metadata.items():
            tag = output_page.tags.get(code)
            if tag is None:
                raise CropError(f"output is missing TIFF metadata tag {code}")
            if int(tag.dtype) != dtype or tag.count != count:
                raise CropError(f"TIFF metadata type/count changed for tag {code}")
            if _raw_payload(output, tag) != payload:
                raise CropError(f"TIFF metadata payload changed for tag {code}")
    with Image.open(output) as output_image:
        output_exif = output_image.getexif()
        for code, expected_ifd in pointers.items():
            if code not in output_image.tag_v2:
                raise CropError(f"output is missing nested TIFF metadata tag {code}")
            actual_ifd = dict(output_exif.get_ifd(code))
            if not _equal_metadata_value(actual_ifd, expected_ifd):
                raise CropError(f"nested TIFF metadata changed for pointer tag {code}")


def _crop_box(
    width: int, height: int, args: argparse.Namespace
) -> tuple[int, int, int, int]:
    if args.box is not None:
        left, top, right, bottom = args.box
    elif args.keep_height is not None:
        left, top, right, bottom = 0, 0, width, args.keep_height
    else:
        left, top = 0, 0
        right, bottom = width, height - args.remove_bottom
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise CropError(
            f"invalid crop ({left}, {top}, {right}, {bottom}) for {width}x{height} image"
        )
    if (left, top, right, bottom) == (0, 0, width, height):
        raise CropError("crop would not remove any pixels")
    return left, top, right, bottom


def crop_one(source: Path, output: Path, args: argparse.Namespace) -> None:
    if source.resolve() == output.resolve():
        raise CropError("input and output paths must differ")
    if output.exists() and not args.overwrite:
        raise CropError(f"output exists (use --overwrite): {output}")

    metadata = _metadata_snapshot(source)
    pointers = _pointer_snapshot(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise CropError(f"only single-page TIFFs are supported: {source}")
        box = _crop_box(*image.size, args)
        if args.dry_run:
            print(f"DRY RUN: {source} -> {output}; {image.size} -> {(box[2]-box[0], box[3]-box[1])}")
            return

        tiffinfo = TiffImagePlugin.ImageFileDirectory_v2()
        exif = image.getexif()
        for code, value in image.tag_v2.items():
            if code in STRUCTURAL_TAGS:
                continue
            if code in POINTER_IFDS:
                child = exif.get_ifd(code)
                if child:
                    tiffinfo[code] = dict(child)
                continue
            if code in image.tag_v2.tagtype:
                tiffinfo.tagtype[code] = image.tag_v2.tagtype[code]
            tiffinfo[code] = value
        cropped = image.crop(box)
        compression = image.info.get("compression", "raw")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            cropped.save(
                temporary,
                format="TIFF",
                tiffinfo=tiffinfo,
                compression=compression,
            )
            _restore_raw_metadata(temporary, metadata)
            _verify(source, temporary, box, metadata, pointers)
            os.replace(temporary, output)
            if not args.no_preserve_times:
                shutil.copystat(source, output)
        finally:
            temporary.unlink(missing_ok=True)

    print(
        f"Wrote {output}: {image.size[0]}x{image.size[1]} -> "
        f"{box[2]-box[0]}x{box[3]-box[1]}; "
        f"verified {len(metadata) + len(pointers)} metadata tags"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop single-page TIFFs and verify preservation of metadata and pixels."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="input .tif/.tiff files")
    geometry = parser.add_mutually_exclusive_group(required=True)
    geometry.add_argument(
        "--keep-height", type=int, metavar="ROWS", help="keep rows 0 through ROWS-1"
    )
    geometry.add_argument(
        "--remove-bottom", type=int, metavar="ROWS", help="remove ROWS rows from the bottom"
    )
    geometry.add_argument(
        "--box",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="crop rectangle; right and bottom are exclusive",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="destination directory (default: beside each input)"
    )
    parser.add_argument("--prefix", default="cropped_", help="output filename prefix")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="show outputs without writing")
    parser.add_argument(
        "--no-preserve-times",
        action="store_true",
        help="do not copy filesystem timestamps and permissions",
    )
    args = parser.parse_args(argv)
    if args.keep_height is not None and args.keep_height <= 0:
        parser.error("--keep-height must be positive")
    if args.remove_bottom is not None and args.remove_bottom <= 0:
        parser.error("--remove-bottom must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    failures = 0
    for source in args.inputs:
        try:
            if not source.is_file():
                raise CropError(f"input does not exist: {source}")
            if source.suffix.lower() not in {".tif", ".tiff"}:
                raise CropError(f"input is not a TIFF file: {source}")
            directory = args.output_dir if args.output_dir is not None else source.parent
            output = directory / f"{args.prefix}{source.name}"
            crop_one(source, output, args)
        except Exception as error:
            failures += 1
            print(f"ERROR: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
