#---------------------------------------------------------------------------#
#                                                                           #
# SEM/BSE 3D surface reconstruction: 2. Backend part                        #
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
# Aided by :                                                                #
#  - GPT4 with CoderPad plugin                                              #
#  - Copilot in VSCode                                                      #
#  - Claude 3.5 Sonnet in cursor.                                           #
#                                                                           #
#---------------------------------------------------------------------------# 

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import radon
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
import vtk
import datetime
from scipy.optimize import curve_fit


pixelsize0 = 1e-6 # default value in meter, if the user asks to search in the TIF file, but there is no PixelWidth in the TIF file


# Configure Matplotlib to use LaTeX for text rendering
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino'] 
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{pxfonts}'

# # Backscattering coefficient models from [1] Böngeler, R., Golla, U., Kässens, M., Reimer, L., Schindler, B., Senkel, R. and Spranck, M., 1993. Electron‐specimen interactions in low‐voltage scanning electron microscopy. Scanning, 15(1), pp.1-18. DOI: https://doi.org/10.1002/sca.4950150102
def backscattering_coefficient(phi, Z):
    return (1+np.cos(phi*np.pi/180))**(-9/np.sqrt(Z))

def backscattering_coefficient_2(phi, Z):
    return 0.89*(backscattering_coefficient(0, Z)/0.89)**np.cos(phi*np.pi/180)

def write_vtk(filename, X, Y, z):
    """
    Save {x, y, z} data as a VTK structured grid and color it by the z-values.

    Args:
        filename (str): Output file name.
        X (numpy.ndarray): 2D array of x-coordinates.
        Y (numpy.ndarray): 2D array of y-coordinates.
        Z (numpy.ndarray): 2D array of z-coordinates.
    """
    # Ensure input arrays are numpy arrays
    X, Y, Z = np.asarray(X), np.asarray(Y), np.asarray(z)

    # Check consistency of dimensions
    if not (X.shape == Y.shape == Z.shape):
        raise ValueError("X, Y, and Z must have the same dimensions")

    # Get dimensions
    ny, nx = X.shape  # Note: VTK uses (nx, ny, nz) ordering

    # Create points for the grid
    points = vtk.vtkPoints()
    for j in range(ny):
        for i in range(nx):
            points.InsertNextPoint(X[j, i], Y[j, i], Z[j, i])

    # Create structured grid
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nx, ny, 1)  # Grid dimensions (nx, ny, nz)
    grid.SetPoints(points)

    # Add z-value as a scalar attribute for coloring
    z_values = vtk.vtkDoubleArray()
    z_values.SetName("Z-Value")
    z_values.SetNumberOfComponents(1)
    z_values.SetNumberOfTuples(nx * ny)

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            z_values.SetValue(idx, Z[j, i])

    grid.GetPointData().SetScalars(z_values)

    # Write to VTK file
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()

# Function that outputs log information both in terminal and log file
def log(logFile, text):
    print("*     "+text)
    logFile.write(text + "\n")

def parabolic_surface(params, X, Y):
    a, b, c = params
    x0 = (np.max(X[0,:]) - np.min(X[0,:]))/2.
    y0 = (np.max(Y[:,0]) - np.min(Y[:,0]))/2.
    return (X-x0)**2/(2*a) + (Y-y0)**2/(2*b) + c
def objective_function(params, X, Y, Z):
    return np.sum((Z - parabolic_surface(params, X, Y))**2)

def remove_outside_central_circle(img):
    # When done in this manner, radon sometimes issues a warning that the image should be zero outside the circle, but it anyway works correctly        
    # Create a coordinate grid
    modified_img = img.copy()
    x, y = np.ogrid[:img.shape[0], :img.shape[1]]
    radius = min(img.shape[0], img.shape[1]) // 2
    center_x = img.shape[0] // 2
    center_y = img.shape[1] // 2
    # Create a mask for values outside the circle
    mask = (x - center_x)**2 + (y - center_y)**2 > radius**2
    # Zero out values outside the circle
    modified_img[mask] = 0
    # Crop the image to the central circle
    img = img[center_x-radius:center_x+radius, center_y-radius:center_y+radius]
    return modified_img

# Extract pixel size from the tif image file (if it is there)
def get_pixel_width(filename):
    with open(filename, 'rb') as file:
        # Go to the end of the file FIXME: removed this line because some SEMs write metadata in the beginning of the file
        # file.seek(-3000, 2)  # Go 3000 characters before the end, adjust if needed
        # Read the last part of the file
        content = file.read().decode('ISO-8859-1')
        TAG_PIXEL_WIDTH = ["PixelWidth=", "XResolution=", "ResolutionX=", "Image Pixel Size = "]

        import re
        for tag in TAG_PIXEL_WIDTH:
            start_index = content.find(tag)
            if start_index != -1:
                start_index += len(tag)
                end_index = content.find('\n', start_index)
                pixel_width_value = content[start_index:end_index].strip()
                print("Tag = " + tag + " Pixel width value = " + pixel_width_value)
                # Try to extract value and units if present
                match = re.search(r'(\d+\.?\d*)\s*([a-zA-Zµ]*)?', pixel_width_value)
                if match:
                    value = float(match.group(1))
                    units = match.group(2).lower() if match.group(2) else "m"  # Default to meters if no unit
                    # Convert to meters based on units
                    if units in ["nm", "nanometer", "nanometers"]:
                        return value * 1e-9
                    elif units in ["um", "µm", "micron", "microns", "micrometer", "micrometers"]:
                        return value * 1e-6
                    elif units in ["mm", "millimeter", "millimeters"]:
                        return value * 1e-3
                    elif units in ["cm", "centimeter", "centimeters"]:
                        return value * 1e-2
                    elif units in ["m", "meter", "meters", ""]:
                        return value
                    else:
                        # If units not recognized, assume meters
                        return value
                else:
                    # If no units found, try to convert the raw value
                    try:
                        return float(pixel_width_value)
                    except ValueError:
                        continue  # Try next tag

        # If none of the tags matched or value could not be parsed
        raise ValueError("No pixel width found in the TIF file, introduce it manually in the interface.")

# #######################################################################################
# Use Frankot and Chellappa method to integrate surface from two gradients
# See: R.T. Frankot, R. Chellappa (1988). Method for enforcing integrability in 
# shape from shading algorithms, IEEE Trans. Pattern Anal. Mach. Intell. 10(4):439-451.
# #######################################################################################
def reconstruct_surface_FFT(Gx, Gy, cutoff = 0):
    # Fourier transform of the gradients
    Cx = np.fft.fft2(Gx) 
    Cy = np.fft.fft2(Gy) 
    n, m = Gx.shape

    # Wave numbers
    kx = np.fft.fftshift(np.arange(0, m) - m / 2) 
    ky = np.fft.fftshift(np.arange(0, n) - n / 2)  
    Kx, Ky = np.meshgrid(kx, ky)

    # Cutoff high-frequency (to remove noise) components if needed
    if cutoff > 0:
        cutoff_sq = (min(m,n)*cutoff/2)**2
        # Create a mask for high frequencies using vectorized operations
        freq_mask = ((Kx**2 + Ky**2) > cutoff_sq) & ((Kx**2 + (Ky-n)**2) > cutoff_sq) & \
                    ((Kx-m)**2 + (Ky-n)**2) > cutoff_sq & ((Kx-m)**2 + Ky**2) > cutoff_sq
        # Apply mask to both Fourier transforms
        Cx[freq_mask] = 0
        Cy[freq_mask] = 0


    # The minimizer is given by the inverse Fourier transform of the solution
    denom = Kx**2 + Ky**2
    C = np.where(denom != 0, -1j * ( Ky * Cx + Kx * Cy) / denom, 0)

    #CAUTION: Do not try to remove curvature in Fourier space, it can produce a lot of artefacts!

    # Return to real space
    Cinv = np.fft.ifft2(C)
    z = np.real(Cinv)

    # Set the mean value to zero
    z = z - np.mean(z)

    return z

# #######################################################################################
# Use direct integration of the surface from two gradients line by line and ensure
# the minimal distance between adjacent lines, in the end average the two surfaces
# #######################################################################################
def reconstruct_surface_direct_integration(Gx, Gy, pixelsize): 
    Gx -= np.mean(Gx)
    Gy -= np.mean(Gy)           
    Gx = Gx * pixelsize
    Gy = Gy * pixelsize
    # Integrate along x-direction
    int_x = np.cumsum(Gx, axis=0)
    int_x_aligned = np.zeros(int_x.shape)
    int_x_aligned[0,:] = 0 #int_x[0,:]
    for j in range(1, int_x.shape[1]):
        int_x_aligned[:,j] = int_x[:,j] + np.mean(int_x_aligned[:,j-1] - int_x[:,j])
    
    # Integrate along y-direction
    int_y = np.cumsum(Gy, axis=1)
    int_y_aligned = np.zeros(int_y.shape)
    int_y_aligned[:,0] = int_x_aligned[:,0]
    for i in range(1, int_y.shape[0]):
        int_y_aligned[i,:] = int_y[i,:] + np.mean(int_y_aligned[i-1,:] - int_y[i,:])
    
    int_x_aligned -= np.mean(int_x_aligned)
    int_y_aligned -= np.mean(int_y_aligned)

    # Assemble the surface by averaging
    # z = 0.5 * (int_x_aligned + int_y_aligned)
    z = int_y_aligned
    
    # Adjust the mean value to zero
    z = z - np.mean(z)
    
    return z

def plot_image_decomposition(imgs, img1, G1, G2, timeStamp, logFile):
    """
    Plot the images: top row is original images, bottom row is polar decomposition
    """
    # Plot the images: top row is original images, bottom row is polar decomposition
    fig,ax = plt.subplots(2,3,figsize=(12,8))
    # remove all ticks and labels
    for axi in ax.flat:
        axi.xaxis.set_visible(False)
        axi.yaxis.set_visible(False)
    # Decrease spacing between subplots
    plt.subplots_adjust(wspace=0., hspace=0.)
    # Decrease margins
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

    plt.subplot(2,3,1)
    plt.imshow(imgs[0])
    plt.title("Image 1, $I_1$")

    plt.subplot(2,3,2)
    plt.imshow(imgs[1])
    plt.title("Image 2, $I_2$")

    plt.subplot(2,3,3)
    plt.imshow(imgs[2])
    plt.title("Image 3, $I_3$")

    plt.subplot(2,3,4)
    plt.imshow(img1)
    plt.title("Principal Image, $A$")

    plt.subplot(2,3,5)
    plt.imshow(G1)
    plt.title("Normalized principal Image 2, $G_1$")

    plt.subplot(2,3,6)
    plt.imshow(G2)
    plt.title("Normalized principal Image 3, $G_2$")

    plt.tight_layout()
    # Save with a unique identifier with a time tag and save in info in log file
    filename = "Images_decomposition" + timeStamp + ".png"
    fig.savefig(filename, dpi=300)
    log(logFile,"Images decomposition saved to " + filename)

def plot_radon_rms(total_angles, total_rms, theta1, theta2, filename=None):
    plt.figure(figsize=(6, 4))
    
    # Find and highlight the minimum RMS angle
    min_angle = theta1
    min_angle2 = theta2
    max_rms = np.max(total_rms)
    total_rms = total_rms/max_rms

    plt.plot(total_angles, total_rms, 'k-', label='Radon RMS')
    plt.scatter(total_angles, total_rms, c='green', marker='o', s=20, zorder=10)
    plt.ylim(None,1.0)
    
    plt.axvline(x=min_angle, color="k", linestyle="--")
    plt.text(min_angle*1.05, 0.5, "$\\theta_1$ = {0:.2f}".format(min_angle), color="k")
    plt.axvline(x=min_angle2, color="k", linestyle="--")
    plt.text(min_angle2*1.05, 0.1, "$\\theta_2$ = {0:.2f}".format(min_angle2), color="k")
    plt.xlim(0,180)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Normalized Radon RMS Value')
    plt.title('Radon Transform RMS vs Angle')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)

def compute_image_gradients(imgs):
    """
    Compute robust gradients using PCA-like approach with proper correlation matrix
    
    Parameters:
    -----------
    imgs : list of numpy.ndarray
        List of input images (same shape)
    
    Returns:
    --------
    tuple: (img1,G1, G2)
        Gradient images computed from PCA decomposition
    """
    # Ensure all images have the same shape
    if len(set(img.shape for img in imgs)) > 1:
        raise ValueError("All input images must have the same shape")
    
    # Reshape images into 2D matrix (each row is a flattened image)
    img_matrix = np.array([img.flatten() for img in imgs])
    
    nb_images = img_matrix.shape[0]
    
    # Compute correlation matrix
    CorrMatrix = np.zeros((nb_images,nb_images))
    for i in range(nb_images):
        for j in range(nb_images):
            CorrMatrix[i,j] = np.dot(img_matrix[i], img_matrix[j]) #/ np.sqrt(np.dot(img_matrix[i], img_matrix[i]) * np.dot(img_matrix[j], img_matrix[j]))
    
    # Perform SVD on the correlation matrix
    U, S, V = np.linalg.svd(CorrMatrix)
        
    # Reconstruct images using the first two principal components
    img_shape = imgs[0].shape
    img1 = np.zeros(img_shape)
    img2 = np.zeros(img_shape)
    img3 = np.zeros(img_shape)
    
    for i in range(nb_images):
        img1 += U[i, 0] * imgs[i]
        img2 += U[i, 1] * imgs[i]
        img3 += U[i, 2] * imgs[i]
    # Avoid division by zero
    meanImg1 = np.mean(img1)
    img1 = np.where(img1 == 0, meanImg1, img1)
    
    # Compute gradients
    # Normalize by the first principal component image
    G1 = img2 / (img1)
    G2 = img3 / (img1)    
        
    return img1, G1, G2


def constructSurface(imgNames, 
                     Plot_images_decomposition, 
                     GaussFilter, 
                     sigma, 
                     ReconstructionMode,
                     RemoveCurvature=False, 
                     cutoff_frequency=0, 
                     save_file_type="", 
                     time_stamp=False, 
                     pixelsize=None, 
                     ZscalingFactorPerPixel=1.0, 
                     Z_ref=None,
                     Z_current=None,
                     logFile=None):
        """
        Construct the surface from the images
        """
        return_message = ""
        # Create a log file with time stamp        
        now = datetime.datetime.now()
        if time_stamp:
            timeStamp = "_"+now.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            timeStamp = ""
        if logFile is None:
            logFileName = "log" + timeStamp + ".log"
            logFile = open(logFileName, "a")        
        log(logFile,"All information is saved to " + logFile.name)
        log(logFile,"Parameters:")
        log(logFile,"   / Plot intermediate images = " + str(Plot_images_decomposition))
        log(logFile,"   / Gauss filter = " + str(GaussFilter))
        log(logFile,"   / STD Gauss filter  = " + str(sigma))
        log(logFile,"   / Remove curvature = " + str(RemoveCurvature))
        log(logFile,"   / Reconstruction Mode = " + str(ReconstructionMode))
        log(logFile,"   / FFT cutoff frequency = " + str(cutoff_frequency))
        log(logFile,"   / Output file type = " + str(save_file_type))
        if time_stamp:
            log(logFile,"   / Time stamp = " + str(time_stamp)[1:])
        else:
            log(logFile,"   / Time stamp = No")
        log(logFile,"   / Pixel size = " + str(pixelsize) +" (m)")
        log(logFile,"   / Z scaling factor per pixel = " + str(ZscalingFactorPerPixel) + " (1/m)")
        ZscalingFactor = ZscalingFactorPerPixel * pixelsize
        log(logFile,"   / Z scaling factor = " + str(ZscalingFactor) + " (-)")
        tilt_sensitivity_factor = 1.0
        if Z_ref is not None and Z_current is not None:
            # # Approximate difference in tilt sensitivity between reference material and current material in the interval 0-20 degrees
            angles = np.linspace(0,20,100)
            tilt_sensitivity_factor = np.mean(backscattering_coefficient_2(angles, Z_ref) / backscattering_coefficient_2(angles, Z_current))
            log(logFile,"   / Z_ref = " + str(Z_ref))
            log(logFile,"   / Z_current = " + str(Z_current))
            log(logFile,"   / Tilt sensitivity factor = " + str(tilt_sensitivity_factor))
        else:
            log(logFile,"   / Z_ref = None")
            log(logFile,"   / Z_current = None")
            log(logFile,"   / Tilt sensitivity factor = " + str(tilt_sensitivity_factor))

        # Save the names of the images to the log file
        log(logFile,"Images folder:" + "/".join(imgNames[0].split("/")[:-1]))
        log(logFile,"Images names:")
        for imgName in imgNames:
            log(logFile,"    / " + imgName.split("/")[-1])
        
# ======================================================================== #
#   1. Read the images, remove the white line at the bottom of the image   #
#   1.2 If required, filter the images with a Gaussian filter              #
# ======================================================================== #
        tmp = plt.imread(imgNames[0])
        # Detect first white line in the image
        cutY = tmp.shape[0]
        for i in range(1,tmp.shape[0]):
            if abs(np.mean(tmp[i,:]) - 1437.) < 2:
                cutY = i-1
                break
        # Save the cutY value to the log file
        log(logFile,"SEM data starts at " + str(cutY))
        imgs = np.zeros((3,cutY, tmp.shape[1]))
        for i in range(3):
            imgs[i] = plt.imread(imgNames[i])[:cutY,:] # removes 

        # Replace nan values with interpolated values
        imgs[0] = np.nan_to_num(imgs[0])
        imgs[1] = np.nan_to_num(imgs[1])
        imgs[2] = np.nan_to_num(imgs[2])

        if GaussFilter:
            # Gauss filter all gradient images
            for i in range(imgs.shape[0]):
                imgs[i] = gaussian_filter(imgs[i], sigma=sigma)
            log(logFile,"Gauss filter with sigma = " + str(sigma) + " is applied to all images.")

# ======================================================================== #
#   2. Construct correlation matrix and compute image gradients            #
#   3. SVD to find the principal components and compute image gradients    #
#   4. Compute image gradients from the principal components               #
# ======================================================================== #
        img1, G1, G2 = compute_image_gradients(imgs)

        if Plot_images_decomposition:
            plot_image_decomposition(imgs, img1, G1, G2, timeStamp, logFile)


# =============================================================================== #
#   5. Use Radon transform to find the angle of the main gradient directions      #
# ============================================================================== #
        # Construct the Radon transform of the gradient images
        # To accelerate do it in an iterative manner and for G1 only
        n_dissection = 10
        angle_start = 0
        angle_end = 180
        iteration = 0
        max_iteration = 4
        
        # Arrays to store all angles and RMS values across iterations
        total_angles = []
        total_rms = []
        
        # Arrays for final values to plot
        Radon_rms = np.zeros(max_iteration*n_dissection)
        Radon_angles = np.zeros(max_iteration*n_dissection)
        
        while True:
            log(logFile,"   / Radon search: iteration " + str(iteration) + " angle_start = " + str(angle_start) + " angle_end = " + str(angle_end))
            theta = np.linspace(angle_start, angle_end, n_dissection, endpoint=False)
            # Radon_angles[iteration*n_dissection:(iteration+1)*n_dissection] = theta
            
            # Put all G1c values zero outside the circle
            G1c = remove_outside_central_circle(G1)
            
            # Compute the Radon transform
            R1 = radon(G1c, theta=theta, circle=True)
            
            # Find the RMS of the angles
            R1t = np.sum((R1 - np.mean(R1, axis=0))**2, axis=0)
            # Radon_rms[iteration*n_dissection:(iteration+1)*n_dissection] = R1t
            
            # Store all angles and RMS values for this iteration
            total_angles.extend(theta)
            total_rms.extend(R1t)
            
            # Find the angle with the minimum RMS
            theta1 = theta[np.argmin(R1t)]
            angle_start = theta1 - 2*(angle_end - angle_start)/n_dissection
            angle_end = theta1 + 2*(angle_end - angle_start)/n_dissection
            iteration += 1
            if iteration > max_iteration:
                break
                
        # Convert lists to numpy arrays and order angles in ascending order
        total_angles = np.array(total_angles)
        total_rms = np.array(total_rms)
        sorted_indices = np.argsort(total_angles)
        total_angles = total_angles[sorted_indices]
        total_rms = total_rms[sorted_indices]

        theta2 = theta1 + 90
        if theta2 > 180:
            theta2 = theta2 - 180
        # Save theta angles in log file
        log(logFile,"theta1 = " + str(theta1))
        log(logFile,"theta2 = " + str(theta2))

        if Plot_images_decomposition:
            filename = "RadonTransformRMS" + timeStamp + ".pdf"
            plot_radon_rms(total_angles, total_rms, theta1, theta2, filename=filename)
            log(logFile,"Radon transform RMS saved to " + filename)

# =================================================================================== #
#   6. Reorient gradients along x and y directions (oriented along theta1 and theta2) #
# =================================================================================== #
        Gx = np.cos(theta1*np.pi/180.)*G1 + np.cos(theta2*np.pi/180.)*G2
        Gy = np.sin(theta1*np.pi/180.)*G1 + np.sin(theta2*np.pi/180.)*G2

        # Remove macroscopic tilt
        Gx -= np.mean(Gx)
        Gy -= np.mean(Gy)

        if Plot_images_decomposition:
            # Create a GridSpec layout with 2 rows and 2 columns
            # The top row will contain the images and the bottom row will contain the colorbars
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.05])

            fig = plt.figure(figsize=(15, 8))

            # First image and its colorbar
            ax1 = fig.add_subplot(gs[0, 0])
            cax1 = fig.add_subplot(gs[1, 0])
            im1 = ax1.imshow(Gx)
            plt.colorbar(im1, cax=cax1, orientation='horizontal')
            ax1.set_title("Gx")

            # Second image and its colorbar
            ax2 = fig.add_subplot(gs[0, 1])
            cax2 = fig.add_subplot(gs[1, 1])
            im2 = ax2.imshow(Gy)
            plt.colorbar(im2, cax=cax2, orientation='horizontal')
            ax2.set_title("Gy")
            plt.tight_layout()

            # Save with a unique identifier with a time tag and save in info in log file
            filename = "Gradients" + timeStamp + ".png"
            fig.savefig(filename, dpi=300)
            log(logFile,"Gradients along x and y directions are saved to " + filename)

# ========================================================= #
#     7. Reconstruct the surface from gradients Gx,Gy       #
# ========================================================= #
        reconstruction_type = ""
        if ReconstructionMode == "FFT":
            z = reconstruct_surface_FFT(Gx, Gy, cutoff_frequency)
            reconstruction_type = "FFT"
        elif ReconstructionMode == "DirectIntegration":
            z = reconstruct_surface_direct_integration(Gx, Gy, pixelsize)
            reconstruction_type = "DirectIntegration"
        else:
            log(logFile,"Error, unknown reconstruction mode")
            logFile.close()
            return None      

        # convert to micrometers and scale according to user-defined scaling
        scalingFactorAndUnits = 1e6 * ZscalingFactor * tilt_sensitivity_factor
        z = scalingFactorAndUnits * z 
        n,m = z.shape
        X,Y = np.meshgrid(np.arange(0, m*pixelsize*1e6, pixelsize*1e6), np.arange(0, n*pixelsize*1e6, pixelsize*1e6))

# ========================================================= #
#          8. Remove curvature defect                       #
# ========================================================= #
        if RemoveCurvature:
            # First, fit parabolas at the edges to get initial guess 
            xlin = X[0,:].copy()
            ylin = Y[:,0].copy()
            ZXlin = z[0,:].copy()
            ZYlin = z[:,0].copy()
            def parabola(x,x0,R,z0):
                return (x-x0)**2/(2*R) + z0
            popt, pconv  = curve_fit(parabola, xlin, ZXlin)
            poptneg, pconvneg  = curve_fit(parabola, xlin, -ZXlin)
            if popt[1] < poptneg[1]:    
                Rx = popt[1]
            else:
                Rx = -poptneg[1]                
            popt, pconv  = curve_fit(parabola, ylin, ZYlin)
            poptneg, pconvneg  = curve_fit(parabola, ylin, -ZYlin)
            if popt[1] < poptneg[1]:
                Ry = popt[1]
            else:
                Ry = -poptneg[1]
            Lx = np.max(X[0,:]) - np.min(X[0,:])
            Ly = np.max(Y[:,0]) - np.min(Y[:,0])

            initial_params = [Rx, Ry, 0.0]
            result = minimize(objective_function, initial_params,
                                args=(X, Y, z),
                                bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)])
            error = objective_function(result.x, X, Y, z)

            # Subtract the fitted parabolic surface
            Rx, Ry, dz = result.x
            P = parabolic_surface(result.x, X, Y)
            z -= P

            if Rx < 0 and Ry < 0:
                log(logFile, f"Curvature was successfully removed: Rx = {Rx:.2f} um, Ry = {Ry:.2f} um, dz = {dz:.2f} um")
            elif Rx * Ry < 0:
                log(logFile, f"WARNING! WARNING! WARNING!\nWarning: Wrong order of images. Image was reconstructed but the results are not reliable. Images should be reordered in a different way: Rx = {Rx:.2f} um, Ry = {Ry:.2f} um, dz = {dz:.2f} um")
                return_message = "Warning: Wrong order of images. The result is not reliable. Reshuffle images and run again."
            else: # if both curvatures are negative, the image should be flipped
                log(logFile, f"Curvatures were successfully removed: Rx = {Rx:.2f} um, Ry = {Ry:.2f} um, dz = {dz:.2f} um")
                log(logFile, "The reconstructed surface is flipped.")
                z *= -1

#========================================================= #
#            Plot the reconstructed surface                #
#========================================================= #

        fig, ax = plt.subplots(figsize=(8,10))
        z = np.rot90(z)
        img = ax.imshow(z, extent=[1e6*X.shape[0]*pixelsize, 0, 0, 1e6*X.shape[1]*pixelsize], interpolation="none")
        ax.set_xlabel(r"$y$, $\mu$m")
        ax.set_ylabel(r"$x$, $\mu$m")
        ax.set_title("Reconstructed surface")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.5)
        cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
        cbar.set_label(r'$z$, $\mu$m')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')

        filename = "Surface_"+reconstruction_type + timeStamp + ".png"
        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        returnImgName = filename
        log(logFile, "Surface reconstructed using " + reconstruction_type + " is saved to " + filename)

        # Plot the surface again in the original orientation in grayscale, keeping aspect ratio
        fig,ax = plt.subplots(figsize=(8,8*X.shape[1]/X.shape[0]))
        ax.imshow(z, cmap='gray')
        ax.axis('off')
        ax.set_aspect('auto')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        filename = "Surface_BW_" + reconstruction_type + timeStamp + ".png"
        fig.savefig(filename, dpi=300)

#========================================================= #
#     Save reconstructed surface in different formats      #
#========================================================= #

        x = np.linspace(0, z.shape[1]*pixelsize*1e6, num = z.shape[1])
        y = np.linspace(0, z.shape[0]*pixelsize*1e6, num = z.shape[0])
        X, Y = np.meshgrid(x, y)
        # save in ASCII file readable by gnuplot with the format # x, y, z and empty lines between each row (if needed)
        if save_file_type == "CSV":
            filename = "Surface" + timeStamp + ".csv"
            log(logFile,"Surface saved to " + filename)
            f = open(filename, "w")
            f.write("# x (um), y (um), z (um)\n")           
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    f.write("{0:.6f},{1:.6f},{2:.6f}\n".format(X[i,j], Y[i,j], z[i,j]))                
                # f.write("\n") # if needed for gnuplot, uncomment
            f.close()
        # Save as npz file
        elif save_file_type == "NPZ":
            filename = "Surface" + timeStamp + ".npz"
            np.savez(filename, X=X, Y=Y, Z=z)
            log(logFile,"Surface saved to " + filename)
        # Save as vts (vtk structured grid) file
        elif save_file_type == "VTK":
            filename = "Surface" + timeStamp + ".vts"
            write_vtk(filename, X, Y, z)
            log(logFile,"Surface saved to " + filename)
        else:
            log(logFile,"Surface is not saved")

        now = datetime.datetime.now()
        log(logFile,"Successfully finished at " + now.strftime("%Y-%m-%d %H:%M:%S") + "\n")

        # Save RMS of the surface
        log(logFile,"RMS of the surface = " + str(np.std(z)))
        logFile.close()


        return returnImgName, X, Y, z, return_message

