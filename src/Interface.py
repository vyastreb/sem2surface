#---------------------------------------------------------------------------#
#                                                                           #
# SEM/BSE 3D surface reconstruction: 1. GUI part                            #
#                                                                           #
# Reconstructor for 3D surface from SEM images from                         #
# at least 3 BSE detectors without knowledge of their orientation           #
#                                                                           #
# The reconstruction relies on PCA decomposition and Radon transform        #
# or direct integration of the gradients                                    #
#                                                                           #                          
# V.A. Yastrebov, CNRS, MINES Paris, Aug, 2023                              # 
# Licence: BSD 3-Clause                                                     #
#                                                                           #
# Code constructed using GPT4 with CoderPad plugin and Copilot in VSCode    #
#                                                                           #
#---------------------------------------------------------------------------#

# For GUI
import tkinter as tk
from tkinter import Button, Canvas, Label, filedialog
from PIL import Image, ImageTk
import os, sys
import tempfile
from sem2surface import constructSurface

pixelsize = 0.5 # microns
Plot_images_decomposition = True
GaussFilter = True
sigma = 1.

def header():
    # printed when running the code
    print("************************************************")
    print("*      SEM/BSE 3D surface reconstruction       *")
    print("* V.A. Yastrebov, CNRS, MINES Paris, Aug, 2023 *")
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


        # Create a list to store the canvases for each detector
        self.detector_canvases = []
        self.filename_labels = []
        self.image_references = []

        # Create frames for each detector
        for i in range(5):
            frame = tk.Frame(self.root, relief=tk.SOLID, borderwidth=2)
            frame.grid(row=0, column=i+1, padx=10, pady=10, sticky=tk.W+tk.E+tk.N+tk.S)

            frame.config(width=50, height=50)
            # Configure the frame for expansion
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_rowconfigure(1, weight=1)
            frame.grid_columnconfigure(0, weight=1)

            # Add header label
            header = tk.Label(frame, text=f"Detector {i+1}", font="Helvetica 12 bold")
            header.pack(pady=5)

            # Create a canvas for the image and add to the frame
            canvas = tk.Canvas(frame, width=150, height=100, relief=tk.SUNKEN, borderwidth=1)
            canvas.pack(pady=5)
            self.detector_canvases.append(canvas)

            filename_label = tk.Label(frame, text="", wraplength=90)  # wraplength wraps the text if it's too long
            filename_label.pack(pady=5)
            self.filename_labels.append(filename_label)

        # Configure column weights
        for i in range(6):  # 5 detector frames + 1 left frame
            self.root.grid_columnconfigure(i, weight=1)
            self.root.grid_rowconfigure(i, weight=1)

        self.canvas = Canvas(root)
        self.canvas.grid(row=1, column=0, columnspan=6, sticky=tk.W+tk.E+tk.N+tk.S)

        # Create a frame for the result canvas
        self.result_frame = tk.Frame(self.root, relief=tk.SOLID, borderwidth=2)
        self.result_frame.grid(row=1, column=0, columnspan=6, sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=10)

        # Configure the result frame to expand and fill available space
        self.result_frame.grid_rowconfigure(0, weight=1)
        self.result_frame.grid_columnconfigure(0, weight=1)

        # Create the result canvas within the result frame
        self.result_canvas = tk.Canvas(self.result_frame)
        self.result_canvas.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)  # Use grid to manage the canvas

        self.root.bind('<Configure>', self.on_resize)
        self.after_id = None

        # Configure column weights
        for i in range(6):  # 5 detector frames + 1 left frame
            self.root.grid_columnconfigure(i, weight=1)

        # Configure row weights
        self.root.grid_rowconfigure(0, weight=0)  # For the detector frames and left frame
        self.root.grid_rowconfigure(1, weight=1)  # For the result canvas

        

    # def __init__(self, root):
    #     self.root = root
    #     self.root.title("SEM-BEM 3D surface reconstruction")
    #     self.filepaths = []

    #     # Create a frame to group the logo and buttons
    #     self.left_frame = tk.Frame(root)
    #     # self.left_frame.grid(row=0, column=0, padx=10, sticky=tk.N)
    #     self.left_frame.grid(row=0, column=0, padx=10, sticky=tk.N+tk.W+tk.E)


    #     # Load and display the logo inside the frame
    #     self.logo_image = ImageTk.PhotoImage(Image.open("logo.png"))
    #     self.logo_label = tk.Label(self.left_frame, image=self.logo_image)
    #     self.logo_label.pack(pady=10)

    #     # Buttons inside the frame
    #     self.upload_button = Button(self.left_frame, text="Upload Files", command=self.upload_files)
    #     self.upload_button.pack(pady=10)

    #     self.run_button = Button(self.left_frame, text="Run 3D constr.", command=self.run, state=tk.DISABLED)  # Initially set to DISABLED
    #     self.run_button.pack(pady=10)

    #     # Create a list to store the canvases for each detector
    #     self.detector_canvases = []
    #     self.filename_labels = []
    #     self.image_references = []

    #     # Create frames for each detector
    #     for i in range(5):
    #         frame = tk.Frame(self.root, relief=tk.SOLID, borderwidth=2)
    #         frame.grid(row=0, column=i+1, padx=10, pady=10, sticky=tk.W+tk.E+tk.N+tk.S)

    #         frame.config(width=50, height=50)
    #         # Configure the frame for expansion
    #         frame.grid_rowconfigure(0, weight=1)  # For the header label
    #         frame.grid_rowconfigure(1, weight=1)  # For the canvas
    #         frame.grid_columnconfigure(0, weight=1)
            
    #         # Add header label
    #         header = tk.Label(frame, text=f"Detector {i+1}", font="Helvetica 12 bold")
    #         header.pack(pady=5)
            
    #         # Create a canvas for the image and add to the frame
    #         canvas = tk.Canvas(frame, width=150, height=100, relief=tk.SUNKEN, borderwidth=1)
    #         canvas.pack(pady=5)
    #         self.detector_canvases.append(canvas)

    #         filename_label = tk.Label(frame, text="", wraplength=90)  # wraplength wraps the text if it's too long
    #         filename_label.pack(pady=5)
    #         self.filename_labels.append(filename_label)

    #     # Configure column weights
    #     for i in range(6):  # 5 detector frames + 1 left frame
    #         self.root.grid_columnconfigure(i, weight=1)
    #         self.root.grid_rowconfigure(i, weight=1)

    #     self.canvas = Canvas(root)
    #     self.canvas.grid(row=1, column=0, columnspan=6, sticky=tk.W+tk.E+tk.N+tk.S)

    #     # Create a frame for the result canvas
    #     self.result_frame = tk.Frame(self.root, relief=tk.SOLID, borderwidth=2)
    #     self.result_frame.grid(row=1, column=0, columnspan=6, sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=10)

    #     # Configure the result frame to expand and fill available space
    #     self.result_frame.grid_rowconfigure(0, weight=1)
    #     self.result_frame.grid_columnconfigure(0, weight=1)

    #     # Create the result canvas within the result frame
    #     self.result_canvas = tk.Canvas(self.result_frame)
    #     self.result_canvas.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)  # Use grid to manage the canvas


    #     # self.result_canvas = tk.Canvas(self.root)
    #     # self.result_canvas.grid(row=1, column=0, columnspan=6, sticky=tk.W+tk.E+tk.N+tk.S, padx=10, pady=10)

    #     self.root.bind('<Configure>', self.on_resize)
    #     self.after_id = None

    #     # Configure column weights
    #     for i in range(6):  # 5 detector frames + 1 left frame
    #         self.root.grid_columnconfigure(i, weight=1)

    #     # Configure row weights
    #     self.root.grid_rowconfigure(0, weight=0)  # For the detector frames and left frame
    #     self.root.grid_rowconfigure(1, weight=1)  # For the result canvas
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


    def run(self):
        imgNames = list(self.original_filepaths)
        imgName = constructSurface(imgNames, pixelsize, Plot_images_decomposition, GaussFilter, sigma)
        self.display_reconstruction(imgName)

if __name__ == "__main__":
    header()
    root = tk.Tk()
    uploader = SEMto3Dinterface(root)
    root.mainloop()

