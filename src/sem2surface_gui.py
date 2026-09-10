"""Tkinter interface for sem2surface."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from sem2surface import construct_surface, get_pixel_width


DEFAULT_Z_SCALE = 2.1727243e2
DEFAULT_GAUSSIAN_SIGMA = 1.0


def _prepare_preview_image(source: Image.Image) -> Image.Image:
    """Return an 8-bit RGB image suitable for display by Tk.

    Pillow clips ``I;16`` SEM images when they are converted directly to RGB.
    Normalizing the finite data range first preserves their visible contrast.
    """
    if source.mode == "F" or source.mode == "I" or source.mode.startswith("I;16"):
        pixels = np.asarray(source, dtype=np.float64)
        finite = np.isfinite(pixels)
        preview = np.zeros(pixels.shape, dtype=np.uint8)
        if finite.any():
            minimum = float(pixels[finite].min())
            maximum = float(pixels[finite].max())
            if maximum > minimum:
                scaled = (pixels[finite] - minimum) * (255.0 / (maximum - minimum))
                preview[finite] = np.clip(scaled, 0, 255).astype(np.uint8)
        return Image.fromarray(preview, mode="L").convert("RGB")
    return source.convert("RGB")


def header() -> None:
    print("************************************************")
    print("*      SEM/BSE 3D surface reconstruction       *")
    print("************************************************")


class SEMto3Dinterface:
    """Desktop interface for reconstruction from three to five images."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SEM/BSE 3D surface reconstruction")
        self.root.minsize(980, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.exit_application)

        self.filepaths: list[Path] = []
        self.image_references: list[ImageTk.PhotoImage | None] = [None] * 5
        self.after_id: str | None = None
        self.worker: threading.Thread | None = None
        self.worker_results: queue.Queue[tuple[str, object]] = queue.Queue()

        self._build_controls()
        self._build_detector_panel()
        self._build_result_panel()
        self._build_information_panel()

        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.bind("<Configure>", self.on_resize)

    def _build_controls(self) -> None:
        self.left_frame = tk.Frame(self.root)
        self.left_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ns")

        button_frame = tk.Frame(self.left_frame)
        button_frame.pack(pady=2)
        self.upload_button = tk.Button(
            button_frame, text="Upload Files", command=self.upload_files
        )
        self.upload_button.grid(row=0, column=0, padx=2, pady=2)
        self.reshuffle_button = tk.Button(
            button_frame,
            text="Reshuffle Images",
            command=self.reshuffle_images,
            state=tk.DISABLED,
        )
        self.reshuffle_button.grid(row=0, column=1, padx=2, pady=2)
        self.run_button = tk.Button(
            button_frame, text="Run 3D reconstruction", command=self.run, state=tk.DISABLED
        )
        self.run_button.grid(row=1, column=0, padx=2, pady=2)
        self.exit_button = tk.Button(
            button_frame, text="Exit", command=self.exit_application
        )
        self.exit_button.grid(row=1, column=1, padx=2, pady=2)

        scale_frame = tk.LabelFrame(
            self.left_frame, text="Z scaling factor per pixel (1/m)", padx=5, pady=5
        )
        scale_frame.pack(pady=3, fill="x")
        self.z_scale_entry = tk.Entry(scale_frame)
        self.z_scale_entry.insert(0, str(DEFAULT_Z_SCALE))
        self.z_scale_entry.pack(fill="x")

        format_frame = tk.LabelFrame(
            self.left_frame, text="Output", padx=5, pady=5
        )
        format_frame.pack(pady=3, fill="x")
        self.output_format = tk.StringVar(value="do not save")
        format_row = tk.Frame(format_frame)
        format_row.pack(fill="x")
        for label, value in (
            ("CSV", "CSV"),
            ("NPZ", "NPZ"),
            ("VTK", "VTK"),
            ("do not save", "do not save"),
        ):
            tk.Radiobutton(
                format_row, text=label, variable=self.output_format, value=value
            ).pack(side=tk.LEFT, padx=(0, 7))

        self.output_directory = tk.StringVar(value=str(Path.cwd()))
        folder_row = tk.Frame(format_frame)
        folder_row.pack(fill="x", pady=(5, 0))
        tk.Button(folder_row, text="Output folder...", command=self.choose_output_folder).pack(
            side=tk.LEFT
        )
        self.output_directory_label = tk.Label(
            folder_row,
            text=self._short_path(Path(self.output_directory.get())),
            anchor="w",
            width=24,
        )
        self.output_directory_label.pack(side=tk.LEFT, padx=(5, 0), fill="x", expand=True)

        filters = tk.Frame(self.left_frame)
        filters.pack(pady=3, fill="x")
        cutoff_frame = tk.LabelFrame(filters, text="FFT cutoff", padx=5, pady=5)
        cutoff_frame.pack(side=tk.LEFT, fill="both", expand=True)
        self.cutoff_slider = tk.Scale(
            cutoff_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Cutoff (%)"
        )
        self.cutoff_slider.pack(fill="x")

        gaussian_frame = tk.LabelFrame(filters, text="Gaussian filter", padx=5, pady=5)
        gaussian_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=(5, 0))
        self.gaussian_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(
            gaussian_frame,
            text="Enable",
            variable=self.gaussian_enabled,
            command=self.toggle_gaussian_entry,
        ).pack(anchor="w")
        self.gaussian_sigma = tk.DoubleVar(value=DEFAULT_GAUSSIAN_SIGMA)
        self.gaussian_entry = tk.Entry(
            gaussian_frame, textvariable=self.gaussian_sigma, width=10, state=tk.DISABLED
        )
        self.gaussian_entry.pack(anchor="w")

        pixel_frame = tk.LabelFrame(
            self.left_frame, text="Pixel size (micrometres)", padx=5, pady=5
        )
        pixel_frame.pack(pady=3, fill="x")
        pixel_row = tk.Frame(pixel_frame)
        pixel_row.pack(fill="x")
        self.use_tiff_pixel_size = tk.BooleanVar(value=True)
        tk.Checkbutton(
            pixel_row,
            text="From TIFF",
            variable=self.use_tiff_pixel_size,
            command=self.toggle_pixel_size_entry,
        ).pack(side=tk.LEFT)
        tk.Label(pixel_row, text="manual").pack(side=tk.LEFT, padx=(10, 2))
        self.pixel_size_entry = tk.Entry(pixel_row, width=12, state=tk.DISABLED)
        self.pixel_size_entry.pack(side=tk.LEFT)

        curvature_frame = tk.LabelFrame(
            self.left_frame, text="Curvature", padx=5, pady=5
        )
        curvature_frame.pack(pady=3, fill="x")
        self.remove_curvature = tk.BooleanVar(value=True)
        tk.Checkbutton(
            curvature_frame,
            text="Remove curvature",
            variable=self.remove_curvature,
            command=self.toggle_curvature_options,
        ).pack(anchor="w")
        self.curvature_mode = tk.StringVar(value="automatic")
        mode_row = tk.Frame(curvature_frame)
        mode_row.pack(anchor="w", padx=(20, 0))
        self.automatic_radio = tk.Radiobutton(
            mode_row,
            text="automatic",
            variable=self.curvature_mode,
            value="automatic",
            command=self.toggle_manual_curvature_entries,
        )
        self.automatic_radio.pack(side=tk.LEFT)
        self.manual_radio = tk.Radiobutton(
            mode_row,
            text="manual",
            variable=self.curvature_mode,
            value="manual",
            command=self.toggle_manual_curvature_entries,
        )
        self.manual_radio.pack(side=tk.LEFT)

        radii_row = tk.Frame(curvature_frame)
        radii_row.pack(anchor="w", padx=(20, 0))
        tk.Label(radii_row, text="Rx (m)").pack(side=tk.LEFT)
        self.rx_entry = tk.Entry(radii_row, width=10, state=tk.DISABLED)
        self.rx_entry.pack(side=tk.LEFT, padx=(2, 8))
        tk.Label(radii_row, text="Ry (m)").pack(side=tk.LEFT)
        self.ry_entry = tk.Entry(radii_row, width=10, state=tk.DISABLED)
        self.ry_entry.pack(side=tk.LEFT, padx=2)

        options_frame = tk.LabelFrame(self.left_frame, text="Options", padx=5, pady=5)
        options_frame.pack(pady=3, fill="x")
        self.timestamp_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(
            options_frame, text="Add time stamp", variable=self.timestamp_enabled
        ).pack(anchor="w")
        self.save_images = tk.BooleanVar(value=False)
        tk.Checkbutton(
            options_frame, text="Save extra images", variable=self.save_images
        ).pack(anchor="w")

    def _build_detector_panel(self) -> None:
        detector_frame = tk.Frame(self.root, width=180)
        detector_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ns")
        self.detector_canvases: list[tk.Canvas] = []
        self.filename_labels: list[tk.Label] = []
        for index in range(5):
            frame = tk.LabelFrame(detector_frame, text=f"Detector {index + 1}")
            frame.pack(pady=2, fill="x")
            canvas = tk.Canvas(frame, width=145, height=82, relief=tk.SUNKEN, borderwidth=1)
            canvas.pack(padx=4, pady=2)
            label = tk.Label(frame, text="", wraplength=150)
            label.pack(padx=2, pady=2)
            self.detector_canvases.append(canvas)
            self.filename_labels.append(label)

    def _build_result_panel(self) -> None:
        result_frame = tk.LabelFrame(self.root, text="Reconstruction")
        result_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.result_canvas = tk.Canvas(result_frame, width=500, height=500)
        self.result_canvas.pack(expand=True, fill="both", padx=5, pady=5)

    def _build_information_panel(self) -> None:
        info_frame = tk.LabelFrame(self.root, text="Information")
        info_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        self.information = tk.Label(info_frame, text="Select three to five detector images.")
        self.information.pack(padx=5, pady=5)

    @staticmethod
    def _short_path(path: Path) -> str:
        text = str(path)
        return text if len(text) <= 32 else "..." + text[-29:]

    def choose_output_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="Select output folder", initialdir=self.output_directory.get()
        )
        if selected:
            self.output_directory.set(selected)
            self.output_directory_label.config(text=self._short_path(Path(selected)))

    def exit_application(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showwarning(
                "Reconstruction running", "Wait for the current reconstruction to finish."
            )
            return
        self.root.destroy()

    def on_resize(self, _event: tk.Event | None = None) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(250, self.update_images)

    def _preview(self, path: Path, width: int, height: int) -> ImageTk.PhotoImage:
        with Image.open(path) as source:
            image = _prepare_preview_image(source)
            image.thumbnail((max(width, 1), max(height, 1)), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def update_images(self) -> None:
        self.after_id = None
        for index, canvas in enumerate(self.detector_canvases):
            canvas.delete("all")
            if index >= len(self.filepaths):
                self.image_references[index] = None
                self.filename_labels[index].config(text="")
                continue
            width = max(canvas.winfo_width() - 6, 1)
            height = max(canvas.winfo_height() - 6, 1)
            try:
                photo = self._preview(self.filepaths[index], width, height)
            except (OSError, ValueError) as exc:
                self.image_references[index] = None
                canvas.create_text(width // 2, height // 2, text="Preview unavailable")
                self.information.config(text=f"Could not preview {self.filepaths[index].name}: {exc}")
            else:
                self.image_references[index] = photo
                canvas.create_image(width // 2, height // 2, anchor=tk.CENTER, image=photo)
            self.filename_labels[index].config(text=self.filepaths[index].name)

    def upload_files(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Select three to five detector images",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        self.filepaths = [Path(path) for path in selected[:5]]
        self.output_directory.set(str(self.filepaths[0].parent))
        self.output_directory_label.config(
            text=self._short_path(self.filepaths[0].parent)
        )
        count = len(self.filepaths)
        valid = 3 <= count <= 5
        self.run_button.config(state=tk.NORMAL if valid else tk.DISABLED)
        self.reshuffle_button.config(state=tk.NORMAL if valid else tk.DISABLED)
        self.information.config(
            text=(
                f"Loaded {count} detector images."
                if valid
                else "At least three detector images are required."
            )
        )
        self.update_images()

    def reshuffle_images(self) -> None:
        count = len(self.filepaths)
        if count == 3:
            self.filepaths[1], self.filepaths[2] = self.filepaths[2], self.filepaths[1]
        elif count in (4, 5):
            self.filepaths = [self.filepaths[-1], *self.filepaths[:-1]]
            self.filepaths[-2], self.filepaths[-1] = (
                self.filepaths[-1],
                self.filepaths[-2],
            )
        self.update_images()

    def display_reconstruction(self, image_path: str | Path) -> None:
        width = max(self.result_canvas.winfo_width() - 10, 1)
        height = max(self.result_canvas.winfo_height() - 10, 1)
        photo = self._preview(Path(image_path), width, height)
        self.result_canvas.delete("all")
        self.result_canvas.create_image(width // 2, height // 2, anchor=tk.CENTER, image=photo)
        self.result_canvas.image = photo

    def toggle_pixel_size_entry(self) -> None:
        state = tk.DISABLED if self.use_tiff_pixel_size.get() else tk.NORMAL
        self.pixel_size_entry.config(state=state)

    def toggle_gaussian_entry(self) -> None:
        state = tk.NORMAL if self.gaussian_enabled.get() else tk.DISABLED
        self.gaussian_entry.config(state=state)

    def toggle_curvature_options(self) -> None:
        state = tk.NORMAL if self.remove_curvature.get() else tk.DISABLED
        self.automatic_radio.config(state=state)
        self.manual_radio.config(state=state)
        self.toggle_manual_curvature_entries()

    def toggle_manual_curvature_entries(self) -> None:
        manual = self.remove_curvature.get() and self.curvature_mode.get() == "manual"
        state = tk.NORMAL if manual else tk.DISABLED
        self.rx_entry.config(state=state)
        self.ry_entry.config(state=state)

    def _read_parameters(self) -> dict[str, object]:
        if len(self.filepaths) < 3:
            raise ValueError("Select at least three detector images")
        z_scale = float(self.z_scale_entry.get())
        if self.use_tiff_pixel_size.get():
            pixel_size = get_pixel_width(self.filepaths[0])
        else:
            manual = float(self.pixel_size_entry.get())
            if manual <= 0:
                raise ValueError("Manual pixel size must be positive")
            pixel_size = manual * 1e-6  # GUI value is in micrometres.

        remove_curvature = self.remove_curvature.get()
        curvature_mode = self.curvature_mode.get()
        manual_rx = manual_ry = None
        if remove_curvature and curvature_mode == "manual":
            manual_rx = float(self.rx_entry.get())
            manual_ry = float(self.ry_entry.get())
            if manual_rx == 0 or manual_ry == 0:
                raise ValueError("Manual curvature radii must be nonzero")

        return {
            "plot_images_decomposition": self.save_images.get(),
            "gaussian_filter_enabled": self.gaussian_enabled.get(),
            "sigma": self.gaussian_sigma.get() if self.gaussian_enabled.get() else 0.0,
            "remove_curvature": remove_curvature,
            "curvature_mode": curvature_mode,
            "manual_rx": manual_rx,
            "manual_ry": manual_ry,
            "cutoff_frequency": self.cutoff_slider.get() / 100.0,
            "save_file_type": self.output_format.get(),
            "time_stamp": self.timestamp_enabled.get(),
            "pixelsize": pixel_size,
            "z_scaling_factor_per_pixel": z_scale,
            "output_dir": Path(self.output_directory.get()),
        }

    def _set_running(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        self.upload_button.config(state=state)
        self.reshuffle_button.config(state=state)
        self.run_button.config(state=state)
        self.exit_button.config(state=state)

    def run(self) -> None:
        try:
            parameters = self._read_parameters()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return
        self._set_running(True)
        self.information.config(text="Reconstruction is running...")
        self.worker = threading.Thread(
            target=self._run_worker, args=(list(self.filepaths), parameters), daemon=True
        )
        self.worker.start()
        self.root.after(100, self._poll_worker)

    def _run_worker(self, paths: list[Path], parameters: dict[str, object]) -> None:
        try:
            result = construct_surface(paths, **parameters)
        except Exception as exc:  # transferred to the UI thread
            self.worker_results.put(("error", exc))
        else:
            self.worker_results.put(("success", result))

    def _poll_worker(self) -> None:
        try:
            status, payload = self.worker_results.get_nowait()
        except queue.Empty:
            if self.worker is not None and self.worker.is_alive():
                self.root.after(100, self._poll_worker)
            return

        self._set_running(False)
        if status == "error":
            self.information.config(text="Reconstruction failed.")
            messagebox.showerror("Reconstruction failed", str(payload))
            return

        image_name, _, _, _, warning = payload
        self.display_reconstruction(image_name)
        output = Path(image_name).parent
        self.information.config(text=f"Reconstruction finished. Output: {output}")
        if warning:
            messagebox.showwarning("Reconstruction warning", warning)


def main() -> None:
    """Launch the desktop application."""
    header()
    root = tk.Tk()
    SEMto3Dinterface(root)
    root.mainloop()


if __name__ == "__main__":
    main()
