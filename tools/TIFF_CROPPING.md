# Metadata-preserving TIFF cropping

`crop_tiff_preserve_metadata.py` crops single-page TIFF images without
discarding microscope metadata. It was developed for the FEI and Zeiss SEM
files used by `sem2surface`.

The program:

1. copies standard, private, and nested EXIF/GPS TIFF metadata;
2. crops the pixels without intensity conversion;
3. restores vendor BYTE/ASCII/UNDEFINED payloads byte-for-byte (important for
   Zeiss tags 34118 and 34119);
4. verifies the pixel crop and every copied metadata payload before atomically
   installing the output; and
5. preserves source filesystem permissions and timestamps by default.

TIFF layout tags such as width, height, strip offsets, and byte counts are
necessarily regenerated for the cropped raster. The source file is never
modified. Existing output files are refused unless `--overwrite` is supplied.

## Requirements

Python 3.10 or newer with `numpy`, `Pillow`, and `tifffile`.

## Examples

Keep the first 512 rows and write `cropped_IMAGE.tif` beside the source:

```bash
python tools/crop_tiff_preserve_metadata.py --keep-height 512 IMAGE.tif
```

Remove a 39-row footer from several files:

```bash
python tools/crop_tiff_preserve_metadata.py --remove-bottom 39 *.tif
```

Apply an explicit `(left, top, right, bottom)` crop and use another directory:

```bash
python tools/crop_tiff_preserve_metadata.py \
  --box 10 20 758 500 --output-dir cropped input.tif
```

Preview the planned filenames and dimensions without writing:

```bash
python tools/crop_tiff_preserve_metadata.py --keep-height 512 --dry-run *.tif
```

Run `python tools/crop_tiff_preserve_metadata.py --help` for all options.

## Scope

The tool deliberately supports single-page TIFFs only. It rejects TIFF image
pyramids and SubIFDs because safely rewriting those structures requires a
separate multi-image workflow. Compression is retained when Pillow supports
the source compression; the SEM files tested here are uncompressed.
