#---------------------------------------------------------------------------#
#                                                                           #
# SEM/BSE 3D surface reconstruction: 1. GUI part                            #
#                                                                           #
# Reconstructor for 3D surface from SEM images from                         #
# at least 3 BSE detectors without knowledge of their orientation           #
#                                                                           #
# The reconstruction relies on SVD-PCA extraction, Radon transform          #
# and Frankot-Chellappa FFT-based reconstruction technique or               #
# direct integration from dz/dx and dz/dy gradients                         #
#                                                                           #                          
# V.A. Yastrebov, CNRS, MINES Paris, Aug 2023-Dec 2024                      #
# Licence: BSD 3-Clause                                                     #
#                                                                           #
# Code constructed using GPT4 with CoderPad plugin and Copilot in VSCode    #
#                                                                           #
#---------------------------------------------------------------------------#

# For GUI
import tkinter as tk
from tkinter import Button, Canvas, Label, filedialog
from PIL import Image, ImageTk
import os
import tempfile
from sem2surface import constructSurface, get_pixel_width, log
import datetime

# Main settings (FIXME to be incorporated into GUI)
# Plot_images_decomposition = True


def header():
    # printed when running the code
    print("************************************************")
    print("*      SEM/BSE 3D surface reconstruction       *")
    print("************************************************")

class SEMto3Dinterface:
    def __init__(self, root):
        # Improved and cleaned up version of the init method
        self.root = root
        self.root.title("SEM-BEM 3D surface reconstruction")
        self.filepaths = []  

        # Create a frame to group the logo and buttons
        self.left_frame = tk.Frame(root)
        self.left_frame.grid(row=0, column=0, padx=10, sticky=tk.N+tk.W+tk.E)

        # Load and display the logo inside the frame
        self.logo_image = ImageTk.PhotoImage(Image.open("logo.png"))
        self.logo_label = tk.Label(self.left_frame, image=self.logo_image)
        self.logo_label.pack(pady=10)

        # Buttons inside the frame
        self.upload_button = Button(self.left_frame, text="Upload Files", command=self.upload_files)
        self.upload_button.pack(pady=10)

        self.run_button = Button(self.left_frame, text="Run 3D constr.", command=self.run, state=tk.DISABLED)  # Initially set to DISABLED
        self.run_button.pack(pady=10)

        # Add exit button
        self.exit_button = Button(self.left_frame, text="Exit", command=self.exit_application)
        self.exit_button.pack(pady=10)


        # Add a frame for output format selection
        self.format_frame = tk.LabelFrame(self.left_frame, text="Output Format", padx=5, pady=5)
        self.format_frame.pack(pady=10, fill="x")

        # Variable to store the selected format
        self.output_format = tk.StringVar(value="CSV")  # Default value

        

        # Radio buttons for format selection
        formats = [("CSV", "CSV"), ("NPZ", "NPZ"), ("VTK", "VTK")]
        for text, value in formats:
            tk.Radiobutton(self.format_frame, 
                          text=text, 
                          variable=self.output_format, 
                          value=value).pack(anchor=tk.W)

        # Create a list to store the canvases for each detector
        self.detector_canvases = []
        self.filename_labels = []
        self.image_references = []

        # Add a frame for FFT cutoff slider
        self.cutoff_frame = tk.LabelFrame(self.left_frame, text="FFT Cutoff", padx=5, pady=5)
        self.cutoff_frame.pack(pady=10, fill="x")

        # Add a slider for FFT cutoff percentage
        self.cutoff_slider = tk.Scale(self.cutoff_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Cutoff (%)")
        self.cutoff_slider.pack(fill="x")

        # Add a frame for pixel size input
        self.pixel_size_frame = tk.LabelFrame(self.left_frame, text="Pixel Size (micrometer)", padx=5, pady=5)
        self.pixel_size_frame.pack(pady=10, fill="x")

        # Add a checkbox for using pixel size from TIFF
        self.use_tiff_pixel_size = tk.BooleanVar(value=True)
        self.tiff_checkbox = tk.Checkbutton(self.pixel_size_frame, text="From TIFF", variable=self.use_tiff_pixel_size, command=self.toggle_pixel_size_entry)
        self.tiff_checkbox.pack(anchor=tk.W)

        # Add an entry for manual pixel size input
        self.pixel_size_entry = tk.Entry(self.pixel_size_frame, state=tk.DISABLED)
        self.pixel_size_entry.pack(fill="x")

        # Add a frame for Reconstruction Mode selection
        self.reconstruction_mode_frame = tk.LabelFrame(self.left_frame, text="Reconstruction Mode", padx=5, pady=5)
        self.reconstruction_mode_frame.pack(pady=10, fill="x")

        # Variable to store the selected reconstruction mode
        self.reconstruction_mode = tk.StringVar(value="FFT")  # Default value

        # Radio buttons for reconstruction mode selection
        modes = [("FFT", "FFT"), ("Direct Integration", "DirectIntegration")]
        for text, value in modes:
            tk.Radiobutton(self.reconstruction_mode_frame, 
                           text=text, 
                           variable=self.reconstruction_mode, 
                           value=value).pack(anchor=tk.W)

        # Add a frame for Gauss filter
        self.gauss_filter_frame = tk.LabelFrame(self.left_frame, text="Gauss Filter", padx=5, pady=5)
        self.gauss_filter_frame.pack(pady=10, fill="x")

        # Add a checkbox for Gauss filter
        self.gauss_filter_enabled = tk.BooleanVar(value=False)
        self.gauss_filter_checkbox = tk.Checkbutton(self.gauss_filter_frame, text="Enable Gauss Filter", variable=self.gauss_filter_enabled, command=self.toggle_gauss_filter_entry)
        self.gauss_filter_checkbox.pack(anchor=tk.W)

        # Add an entry for Gauss filter value
        self.gauss_filter_value = tk.DoubleVar(value=1.)
        self.gauss_filter_entry = tk.Entry(self.gauss_filter_frame, textvariable=self.gauss_filter_value, state=tk.DISABLED)
        self.gauss_filter_entry.pack(fill="x")

        # Add a frame for Remove Curvature option
        self.curvature_frame = tk.LabelFrame(self.left_frame, text="Options", padx=5, pady=5)
        self.curvature_frame.pack(pady=10, fill="x")

        # Add checkbox to add time stamp to the output file name
        self.timestamp_enabled = tk.BooleanVar(value=False)
        self.timestamp_checkbox = tk.Checkbutton(self.curvature_frame, text="Add Time Stamp", variable=self.timestamp_enabled)
        self.timestamp_checkbox.pack(anchor=tk.W)

        # Add a checkbox for Remove Curvature
        self.remove_curvature = tk.BooleanVar(value=True)
        self.curvature_checkbox = tk.Checkbutton(self.curvature_frame, text="Remove Curvature", variable=self.remove_curvature)
        self.curvature_checkbox.pack(anchor=tk.W)

        # Add a checkbox for "Save images"
        self.save_images = tk.BooleanVar(value=True)
        self.save_images_checkbox = tk.Checkbutton(self.curvature_frame, text="Save images", variable=self.save_images)
        self.save_images_checkbox.pack(anchor=tk.W)

        # Create frames for each detector
        detector_frame = tk.Frame(self.root, width=200)  # Set fixed width
        detector_frame.grid(row=0, column=1, rowspan=5, padx=10, pady=10, sticky=tk.W+tk.E+tk.N+tk.S)    
        detector_frame.grid_propagate(False)  # Prevent frame from resizing

        # Configure column weights with absolute sizing
        self.root.grid_columnconfigure(0, weight=0)  # First column (parameters) fixed width
        self.root.grid_columnconfigure(1, weight=0, minsize=200)  # Detector column with minimum width
        self.root.grid_columnconfigure(2, weight=1)  # Result column gets expanding space        

        for i in range(5):
            frame = tk.Frame(detector_frame, relief=tk.SOLID, borderwidth=2)
            frame.pack(pady=5, fill=tk.X)

            # Add header label
            header = tk.Label(frame, text=f"Detector {i+1}", font="Helvetica 12 bold")
            header.pack(pady=5)

            # Create a canvas for the image and add to the frame
            canvas = tk.Canvas(frame, width=150, height=100, relief=tk.SUNKEN, borderwidth=1)
            canvas.pack(pady=5)
            self.detector_canvases.append(canvas)

            filename_label = tk.Label(frame, text="", wraplength=150)
            filename_label.pack(pady=5)
            self.filename_labels.append(filename_label)

            self.canvas = Canvas(root)
            self.canvas.grid(row=0, column=2, rowspan=5, sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=10)
        # Create a frame for the result canvas with fixed 500px width
        self.result_frame = tk.Frame(self.root, width=500, height=400, relief=tk.SOLID, borderwidth=2)
        self.result_frame.grid(row=0, column=2, rowspan=5, sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=10)
        self.result_frame.grid_propagate(False)  # Prevent frame from resizing
        self.result_frame.pack_propagate(False)  # Prevent frame from resizing

        # Add header
        header = tk.Label(self.result_frame, text="Reconstruction", font="Helvetica 12 bold")
        header.pack(pady=5)

        # Create the result canvas within the result frame
        self.result_canvas = tk.Canvas(self.result_frame, width=500, height=350)
        self.result_canvas.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        self.root.bind('<Configure>', self.on_resize)
        self.after_id = None

        # Configure column weights
        self.root.grid_columnconfigure(0, weight=1)  # Detector column
        self.root.grid_columnconfigure(1, weight=2)  # Result column gets more space

        # Configure row weights
        for i in range(5):
            self.root.grid_rowconfigure(i, weight=1)
        self.root.grid_rowconfigure(5, weight=0)  # Result row

    
    
    def exit_application(self):
        self.root.destroy()
        self.root.after(10, self.root.quit)
        

    def on_resize(self, event=None):
        # Cancel the previous scheduled call if it exists
        if self.after_id:
            self.root.after_cancel(self.after_id)

        # Schedule the update_images function to be called after a delay (e.g., 500 milliseconds)
        self.after_id = self.root.after(500, self.update_images)

    def update_images(self, event=None):
        self.root.update_idletasks()  # Update the window to get correct dimensions
        for index, filepath in enumerate(self.filepaths):  # Assuming self.filepaths stores the paths of the uploaded files
            # Get the frame's dimensions
            canvas = self.detector_canvases[index]
            frame_width = canvas.winfo_width() - 4  # Subtracting 2 times the margin (3px on each side)
            frame_height = canvas.winfo_height() - 4

            # Open the image
            image = Image.open(filepath)

            # Calculate the aspect ratio
            aspect_ratio = image.size[0] / image.size[1]

            # Determine the target width and height based on the aspect ratio
            if aspect_ratio > 1:
                # Image is wider than tall
                target_width = frame_width
                target_height = int(frame_width / aspect_ratio)
            else:
                # Image is taller than wide or square
                target_height = frame_height
                target_width = int(frame_height * aspect_ratio)

            # Resize the image
            image = image.resize((target_width, target_height))
            photo = ImageTk.PhotoImage(image)

            # Update the image on the canvas
            canvas.create_image(frame_width // 2, frame_height // 2, anchor=tk.CENTER, image=photo)
            self.image_references[index] = photo  # Update the reference to the new photo image

            # Update result_canvas
         # Update result_canvas
        if hasattr(self, 'result_image_path'):
            # Load the result image
            result_image = Image.open(self.result_image_path)

            # Get the canvas dimensions
            canvas_width = self.result_canvas.winfo_width()
            canvas_height = self.result_canvas.winfo_height()

            # Calculate the aspect ratio of the image
            aspect_ratio = result_image.size[0] / result_image.size[1]

            # Compute scaling factors for width and height
            width_scale = canvas_width / result_image.size[0]
            height_scale = canvas_height / result_image.size[1]

            # Use the smaller of the two scaling factors to ensure the image fits within the canvas
            scale_factor = min(width_scale, height_scale)

            # Determine the target width and height
            target_width = int(result_image.size[0] * scale_factor)
            target_height = int(result_image.size[1] * scale_factor)

            # Resize the image
            result_image = result_image.resize((target_width, target_height))
            result_photo = ImageTk.PhotoImage(result_image)

            # Display the image on the canvas
            self.result_canvas.create_image((canvas_width - target_width) // 2, (canvas_height - target_height) // 2, anchor=tk.NW, image=result_photo)
            self.result_canvas.image = result_photo  # Keep a reference to avoid garbage collection

    def upload_files(self):
        self.filepaths = list(filedialog.askopenfilenames(title="Select up to 5 files", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.tif")], multiple=True))
        self.original_filepaths = self.filepaths
        if self.filepaths != None:
            self.run_button.config(state=tk.NORMAL)   
        
        # Limit to 5 files
        self.filepaths = self.filepaths[:5]

        # Clear the canvas
        self.canvas.delete("all")

        # Display all images
        for index, filepath in enumerate(self.filepaths):
            if filepath.lower().endswith('.tif'):
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                os.system(f"convert {filepath} -geometry 25% {tmp_file.name} > /dev/null 2>&1")
                display_path = tmp_file.name
                self.filepaths[index] = display_path
            else:
                display_path = filepath

            try:
                # Use the appropriate canvas from the list
                canvas = self.detector_canvases[index]
                # Get the frame's dimensions
                frame_width = canvas.winfo_width() - 6  # Subtracting 2 times the margin (3px on each side)
                frame_height = canvas.winfo_height() - 6

                # Open the image
                image = Image.open(display_path)

                # Calculate the aspect ratio
                aspect_ratio = image.size[0] / image.size[1]

                # Determine the target width and height based on the aspect ratio
                if aspect_ratio > 1:
                    # Image is wider than tall
                    target_width = frame_width
                    target_height = int(frame_width / aspect_ratio)
                else:
                    # Image is taller than wide or square
                    target_height = frame_height
                    target_width = int(frame_height * aspect_ratio)

                # Resize the image
                image = image.resize((target_width, target_height))
                photo = ImageTk.PhotoImage(image)                
                canvas.create_image(frame_width // 2, frame_height // 2, anchor=tk.CENTER, image=photo)
                self.image_references.append(photo)




                # image = Image.open(display_path)
                # height = 100
                # width = int(height * image.size[0] / image.size[1])
                # image = image.resize((width, height))
                # photo = ImageTk.PhotoImage(image)
                
                # # Use the appropriate canvas from the list
                # canvas = self.detector_canvases[index]
                # canvas.create_image(75, 50, anchor=tk.CENTER, image=photo)
                # self.image_references.append(photo)
            # If a temporary file was used, delete it
                # if display_path != filepath:
                #     print(display_path)
                #     # replace filepath in self.filepaths by display_path
                #     self.filepaths[index] = display_path
                #     # os.remove(display_path)                
            except Exception as e:
                print(f"Error loading image {filepath}: {e}")

            # Update the filename label
            filename = os.path.basename(filepath)
            self.filename_labels[index].config(text=filename)

        self.update_images()

    def display_reconstruction(self, image_path):
        # Load the image
        image = Image.open(image_path)

        # Get the canvas dimensions
        canvas_width = self.result_canvas.winfo_width()
        canvas_height = self.result_canvas.winfo_height()

        # Calculate the aspect ratio
        aspect_ratio = image.size[0] / image.size[1]

        # Compute scaling factors for width and height
        width_scale = canvas_width / image.size[0]
        height_scale = canvas_height / image.size[1]

        # Use the smaller of the two scaling factors to ensure the image fits within the canvas
        scale_factor = min(width_scale, height_scale)

        # Determine the target width and height
        target_width = int(image.size[0] * scale_factor)
        target_height = int(image.size[1] * scale_factor)

        # Resize the image
        image = image.resize((target_width, target_height))
        photo = ImageTk.PhotoImage(image)

        # Display the image on the canvas
        self.result_canvas.create_image((canvas_width - target_width) // 2, (canvas_height - target_height) // 2, anchor=tk.NW, image=photo)
        self.result_canvas.image = photo  # Keep a reference to avoid garbage collection

    def toggle_pixel_size_entry(self):
        # Enable or disable the pixel size entry based on the checkbox
        if self.use_tiff_pixel_size.get():
            self.pixel_size_entry.config(state=tk.DISABLED)
        else:
            self.pixel_size_entry.config(state=tk.NORMAL)

    def toggle_gauss_filter_entry(self):
        # Enable or disable the Gauss filter entry based on the checkbox
        if self.gauss_filter_enabled.get():
            self.gauss_filter_entry.config(state=tk.NORMAL)
        else:
            self.gauss_filter_entry.config(state=tk.DISABLED)

    def run(self):
        imgNames = list(self.original_filepaths)
        # Get the cutoff percentage from the slider
        cutoff_percentage = self.cutoff_slider.get()

        # Determine the value of Plot_images_decomposition based on the checkbox
        Plot_images_decomposition = self.save_images.get()

        # If time stamp is enabled, create a log file
        if self.timestamp_enabled.get():
            timeStamp = "_"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            timeStamp = ""
        logFileName = "log" + timeStamp + ".log"
        logFile = open(logFileName, "w")        

        if self.use_tiff_pixel_size.get():
            pixelsize = get_pixel_width(imgNames[0])
            log(logFile,"Read from TIF file, PixelWidth = " + str(pixelsize)+ " (m)")
        else:
            pixelsize = float(self.pixel_size_entry.get()) * 1e-6  # Convert micrometers to meters
            log(logFile,"PixelWidth = " + str(pixelsize)+ " (m)")

        # Get the Gauss filter settings
        gauss_filter = self.gauss_filter_enabled.get()
        gauss_sigma = self.gauss_filter_value.get() if gauss_filter else 0

        imgName = constructSurface(imgNames, 
                                   Plot_images_decomposition, 
                                   gauss_filter,  # Use the Gauss filter setting
                                   gauss_sigma,   # Use the Gauss filter value
                                   self.reconstruction_mode.get(),  # Use the selected reconstruction mode
                                   self.remove_curvature.get(),  # Use the Remove Curvature setting
                                   cutoff_frequency=0.01*cutoff_percentage,
                                   save_file_type=self.output_format.get(),
                                   time_stamp=self.timestamp_enabled.get(),
                                   pixelsize=pixelsize, # put pixelsize in meters
                                   logFile=logFile)
        self.display_reconstruction(imgName)

if __name__ == "__main__":
    header()
    root = tk.Tk()
    uploader = SEMto3Dinterface(root)
    root.mainloop()

