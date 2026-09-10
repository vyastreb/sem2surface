# Repository instructions for coding agents

These instructions apply to the entire repository.

## Project purpose

`sem2surface` reconstructs a 3D surface from three to five multi-detector
SEM/BSE images. The supported reconstruction path is principal-image
decomposition followed by Radon orientation and Frankot-Chellappa FFT
integration.

## Product decisions

- Support three to five detector images of identical shape.
- Keep FFT reconstruction as the only reconstruction method.
- Keep the calibrated Z scaling factor; do not restore atomic-number scaling.
- Direct integration is obsolete. The legacy API may reject it explicitly but
  it must not be offered by the CLI or GUI.
- The GUI accepts TIFF, PNG, JPEG, and BMP input.
- Pixel size is expressed in micrometres in the GUI and CLI, metres in the
  Python API, and micrometres in returned coordinate arrays.
- Preserve the compatibility wrapper `constructSurface` unless a deliberate
  breaking release removes it.

## Source layout

- `src/sem2surface.py`: reconstruction core and compatibility API.
- `src/sem2surface_gui.py`: Tkinter desktop interface.
- `src/sem2surface_cli.py`: command-line entry point.
- `examples/`: reference reconstructions and Vickers scaling workflow.
- `tests/`: unit, CLI, GUI-preview, and reference-data tests.
- `pyproject.toml`: package metadata, dependencies, and entry points.

The installed commands are `sem2surface` and `sem2surface-gui`. Setuptools is
configured with a `src` layout and three top-level Python modules rather than a
package directory.

## Portability requirements

- Use `pathlib.Path`; do not construct paths with hard-coded `/` or `\\`.
- Do not introduce shell commands for image processing.
- ImageMagick and LaTeX must not be runtime requirements.
- Keep Matplotlib's non-interactive `Agg` backend in the reconstruction core.
- VTK is optional and must be imported lazily only when VTK output is requested.
- Tkinter work must remain on the main thread. Long reconstruction work belongs
  in the GUI worker thread, with results returned through its queue.
- Preserve strong contrast in high-bit-depth SEM previews. Pillow directly
  clips `I;16` TIFF data when converting to RGB, so normalize it for display;
  never apply that preview normalization to reconstruction input data.

## Numerical compatibility

- Preserve the established FFT normalization for even image sizes while also
  supporting odd dimensions.
- The historical asymmetric Radon narrowing trajectory is intentional because
  the calibration examples were generated with it. Do not change it without
  updating and scientifically validating the reference results.
- Keep SVD signs deterministic across BLAS implementations.
- Do not overwrite example reference outputs during tests. Write generated test
  data to pytest temporary directories or `/tmp`.

## Development and verification

Use a virtual environment. The workstation's system Python may contain an old
Debian/Ubuntu pip and setuptools combination that builds modern PEP 621 projects
as `UNKNOWN-0.0.0`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q
python -m build
python -m twine check dist/*
git diff --check
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`.venv\Scripts\activate.bat` in Command Prompt. Use `py -m venv .venv` when
the `python` command is not available.

Test numerical changes with the real examples as well as synthetic arrays. GUI
changes should receive a non-interactive unit test where possible and a Tk
startup smoke test on a virtual display on Linux.

## Packaging and releases

- The release version is currently duplicated in `pyproject.toml` and
  `src/sem2surface.py`; update both together.
- Build both the source archive and the universal wheel with `python -m build`.
- Validate both files with Twine and test installation in a fresh environment.
- PyPI release files are immutable. Increment the version before rebuilding a
  release that has already been uploaded.
- Never upload to TestPyPI or PyPI, create a token, or modify publishing
  credentials unless the user explicitly requests that external action.
- Never put a PyPI token in the repository, a command argument, a log, or chat.
- Prefer PyPI Trusted Publishing for automated GitHub releases. Manual releases
  should use Twine's interactive credential prompt.

PyPI renders the packaged `README.md`, not the live GitHub file. All README
images must therefore use an absolute HTTPS URL such as
`https://raw.githubusercontent.com/vyastreb/sem2surface/master/img/example.jpg`.
README links to repository files must likewise be absolute GitHub URLs. Run
`python -m twine check dist/*` after every README change.

## Configuration-file safety

- Treat configuration files and dotfiles as important user data.
- Before modifying an existing configuration file, read it completely and
  confirm the exact target.
- Before a material configuration edit, create a timestamped adjacent backup
  and report it, unless the user explicitly declines.
- Make the smallest idempotent patch and preserve unrelated content.
- Never replace or truncate a possible existing configuration file merely
  because a search did not find the expected text.
- Do not reload a shell or another running configuration consumer unless the
  user explicitly requests it.

## Worktree and data safety

The repository may contain unrelated modified and untracked research data.
Preserve it. Do not clean the worktree, delete generated scientific results, or
rewrite example data unless the user explicitly identifies those targets.
