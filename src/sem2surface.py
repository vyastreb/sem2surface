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
# Code constructed using GPT4 with CoderPad plugin and Copilot in VSCode    #
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

pixelsize0 = 1e-6 # default value in meter, if the user asks to search in the TIF file, but there is no PixelWidth in the TIF file
ZscalingFactor = 1./11546.488204918922 # 400 / 3.25521e-07 # Rather random z-scaling factor, normally should be extracted from known topography, e.g. indenter imprint # FIXME bring it to the GUI

# Configure Matplotlib to use LaTeX for text rendering
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino'] 
plt.rcParams['text.usetex'] = True
# Set the font to pxfonts
plt.rcParams['text.latex.preamble'] = r'\usepackage{pxfonts}'

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
    x0 = X.shape[1]/2.
    y0 = X.shape[0]/2.
    return 0.5*(((X-x0)/a)**2 + ((Y-y0)/b)**2) + c
def objective_function(params, X, Y, Z):
    return np.sum((Z - parabolic_surface(params, X, Y))**2)

def VickerIndenter(X,Y,X0,Y0,R,rot,depth,scale):
    angle = 136 * np.pi / 180.
    Xprime = (X-X0)*np.cos(rot) - (Y-Y0)*np.sin(rot)
    Yprime = (X-X0)*np.sin(rot) + (Y-Y0)*np.cos(rot)
    Z = np.zeros(X.shape)

    mask = (X-X0)**2 + (Y-Y0)**2 > R**2

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if Xprime[i,j] >= 0 and abs(Yprime[i,j]) < Xprime[i,j]:
                Z[i,j] = depth + (Xprime[i,j])*np.tan(angle/2.)
            elif Xprime[i,j] < 0 and abs(Yprime[i,j]) < abs(Xprime[i,j]):
                Z[i,j] = depth - (Xprime[i,j])*np.tan(angle/2.)
            elif Yprime[i,j] >= 0 and abs(Xprime[i,j]) < abs(Yprime[i,j]):
                Z[i,j] = depth + (Yprime[i,j])*np.tan(angle/2.)
            elif Yprime[i,j] < 0 and abs(Xprime[i,j]) < abs(Yprime[i,j]):
                Z[i,j] = depth -(Yprime[i,j])*np.tan(angle/2.)
            # else:
            #     Z[i,j] = depth-(Yprime[i,j])*np.tan(angle/2.)
            # else:
            #     Z[i,j] = depth + R/np.sqrt(2)*np.tan(angle/2.)

    Z[mask] = np.nan
    Z *= scale
    return Z

# Extract pixel size from the tif image file (if it is there)
def get_pixel_width(filename):
    with open(filename, 'rb') as file:
        # Go to the end of the file
        file.seek(-3000, 2)  # Go 200 characters before the end, adjust if needed
        # Read the last part of the file
        content = file.read().decode('ISO-8859-1')
        
        # Find the PixelWidth value
        pixel_width_str = "PixelWidth="
        start_index = content.find(pixel_width_str)
        
        if start_index != -1:
            start_index += len(pixel_width_str)
            end_index = content.find('\n', start_index)
            pixel_width_value = content[start_index:end_index].strip()
            return float(pixel_width_value)
        else:
            return None

# #######################################################################################
# Use Frankot and Chellappa method to integrate surface from two gradients
# See: R.T. Frankot, R. Chellappa (1988). Method for enforcing integrability in 
# shape from shading algorithms, IEEE Trans. Pattern Anal. Mach. Intell. 10(4):439-451.
# #######################################################################################
def reconstruct_surface_FFT(Gx, Gy, pixelsize, cutoff=0):
    # Fourier transform of the gradients
    Cx = np.fft.fft2(Gx) 
    Cy = np.fft.fft2(Gy) 

    n, m = Gx.shape

    # # Create a 2D Hamming window (if needed, I do not see any changes)
    # window_1D_n = np.hamming(n)
    # window_1D_m = np.hamming(m)
    # window_2D = np.outer(window_1D_n, window_1D_m)
    # # Apply the window to the image
    # Gx = Gx * window_2D
    # Gy = Gy * window_2D

    # Wave numbers
    kx = np.fft.fftshift(np.arange(0, m) - m / 2) 
    ky = np.fft.fftshift(np.arange(0, n) - n / 2)  
    Kx, Ky = np.meshgrid(kx, ky)
    # Cutoff high-frequency components if needed
    if cutoff > 0:    
        cutoff_sq = (min(m,n)*cutoff/2)**2
        for i in range(m):  
            for j in range(n):
                if i**2 + j**2 > cutoff_sq and (i-m)**2 + j**2 > cutoff_sq and \
                (i-m)**2 + (j-n)**2 > cutoff_sq and i**2 + (j-n)**2 > cutoff_sq:
                        Cx[j, i] = 0
                        Cy[j, i] = 0

    # The minimizer is given by the inverse Fourier transform of the solution
    denom = Kx**2 + Ky**2
    C = np.where(denom != 0, -1j * ( Ky * Cx + Kx * Cy) / denom, 0)

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

def constructSurface(imgNames, Plot_images_decomposition, GaussFilter, sigma, ReconstructionMode,RemoveCurvature=False, cutoff_frequency=0, save_file_type="", time_stamp=False, pixelsize=None, logFile=None):
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

        # Save the names of the images to the log file
        log(logFile,"Image names:")
        for imgName in imgNames:
            log(logFile,imgName)

        # if pixelsize is None:
        #     pixelsize = get_pixel_width(imgNames[0])
        #     if pixelsize:
        #         log(logFile,"Read from TIF file, PixelWidth = " + str(pixelsize)+ " (m)")
        #     else:
        #         log(logFile,"!!! Warning !!! PixelWidth is not found in the tif file, it is set to default value <"+str(pixelsize0)+"> (m). Need to set it manually.")
        #     pixelsize = pixelsize0
        

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
                for j in range(imgs.shape[1]):
                    imgs[i,j] = gaussian_filter(imgs[i,j], sigma=sigma)

        # Construct correlation matrix between images imgs[0], imgs[1], and imgs[2]
        N = len(imgs)
        CorrMatrix = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                CorrMatrix[i,j] = np.dot(imgs[i].flatten(), imgs[j].flatten())

        # Do polar decomposition of the correlation matrix
        U, S, V = np.linalg.svd(CorrMatrix)
        
        img1 = np.zeros(imgs[0].shape)
        img2 = np.zeros(imgs[0].shape)
        img3 = np.zeros(imgs[0].shape)

        for i in range(imgs.shape[0]):
            img1 += U[i,0]*imgs[i]
            img2 += U[i,1]*imgs[i]
            img3 += U[i,2]*imgs[i]
        # To avoid eventual division by zero
        meanImg1 = np.mean(img1)
        img1 = np.where(img1 == 0, meanImg1, img1)

        if Plot_images_decomposition:
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
            plt.imshow(img2/img1)
            plt.title("Normalized principal Image 2, $G_1$")

            plt.subplot(2,3,6)
            plt.imshow(img3/img1)
            plt.title("Normalized principal Image 3, $G_2$")

            plt.tight_layout()
            # Save with a unique identifier with a time tag and save in info in log file
            filename = "Images_decomposition_" + timeStamp + ".png"
            fig.savefig(filename, dpi=300)
            log(logFile,"Images decomposition saved to " + filename)

        # Gradient along unknown directions are given by
        G1 = img2/img1
        G2 = img3/img1

        ##################################################################
        # Use Radon method to find the angle of the unknown direction
        ##################################################################
        # Construct the Radon transform of the gradient images
        # To accelerate do it for G1 only
        theta = np.linspace(0., 180., 180, endpoint=False)
        # Put all G1c and G2c values zero outside the circle

        # minDim = min(G1.shape[0], G1.shape[1])
        # if minDim % 2 == 1:
        #     minDim -= 1
        # G1c = np.array([])
        # if G1.shape[0] > G1.shape[1]:
        #     startWith = int((G1.shape[0]-minDim)/2)
        #     G1c = G1[startWith:startWith+minDim,0:minDim]
        # else:
        #     startWith = int((G1.shape[1]-minDim)/2)
        #     G1c = G1[0:minDim,startWith:startWith+minDim]
        # # Create a coordinate grid
        # x, y = np.ogrid[:minDim, :minDim]
        # center = minDim // 2
        # # Create a mask for values outside the circle
        # mask = (x - center)**2 + (y - center)**2 > (minDim/2)**2
        # # Zero out values outside the circle
        # G1c[mask] = 0


        # When done in this manner, radon sometimes issues a warning that the image should be zero outside the circle, but it anyway works correctly        
        G1c = np.zeros(G1.shape)
        # G2c = np.zeros(G2.shape)
        xc = G1.shape[0]/2
        yc = G1.shape[1]/2
        rad = min(G1.shape[0],G1.shape[1])/2
        for i in range(G1.shape[0]):
            for j in range(G1.shape[1]):
                if (i - xc)**2 + (j - yc)**2 < rad**2:
                    G1c[i,j] = G1[i,j]
                    # G2c[i,j] = G2[i,j]


        R1 = radon(G1c, theta=theta, circle=True)
        # R2 = radon(G2c, theta=theta, circle=True)

        # Find the angle of the unknown direction
        fig,ax = plt.subplots()
        plt.xlim([0, 180])
        plt.xlabel("Projection angle (deg)")
        plt.ylabel("RMS of Radon transform")

        R1t = np.sum((R1 - np.mean(R1, axis=0))**2, axis=0)
        # R2t = np.sum((R2 - np.mean(R2, axis=0))**2, axis=0)

        # plt.ylim([0, 1.05*max(np.max(R1t),np.max(R2t))])
        plt.ylim([0, 1.05*np.max(R1t)])

        plt.plot(theta, R1t, label="R1")
        # plt.plot(theta, R2t, label="R2")
        plt.axhline(y=0, color="k", linestyle="--")
        plt.legend(loc="best")
        
        # Angles corresponding to zero R1t and R2t
        theta1 = theta[np.argmin(R1t)]
        # theta2 = theta[np.argmin(R2t)]
        # if theta2 - theta1 > 0:
        #     ninty = 90
        # else:
        #     ninty = -90
        # theta2_ = theta1 + ninty
        # theta1_ = theta2 - ninty
        # theta1 = (theta1 + theta1_)/2.
        # theta2 = (theta2 + theta2_)/2.
        theta2 = theta1 + 90

        # Save theta angles in log file
        log(logFile,"theta1 = " + str(theta1))
        log(logFile,"theta2 = " + str(theta2))

        plt.axvline(x=theta1, color="r", linestyle="--")
        plt.text(theta1*0.95, 0.4e6, "$\\theta_1$ = {0:.1f}".format(theta1), color="r")
        plt.axvline(x=theta2, color="b", linestyle="--")
        plt.text(theta2*0.95, 0.4e6, "$\\theta_2$ = {0:.1f}".format(theta2), color="b")
        # Save with a unique identifier with a time tag and save in info in log file
        filename = "RadonTransformRMS_" + timeStamp + ".pdf"
        fig.savefig(filename)
        log(logFile,"Radon transform RMS saved to " + filename)

        # Compute gradients along x and y directions
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
            filename = "Gradients_" + timeStamp + ".png"
            fig.savefig(filename, dpi=300)
            log(logFile,"Gradients along x and y directions are saved to " + filename)

        reconstruction_type = ""
        if ReconstructionMode == "FFT":
            z = reconstruct_surface_FFT(Gx, Gy, pixelsize, cutoff=cutoff_frequency)
            reconstruction_type = "FFT"
            # z *= ZscalingFactor * pixelsize # Rather random z-scaling factor
            z *= ZscalingFactor * pixelsize / (6.51042e-08)
        elif ReconstructionMode == "DirectIntegration":
            z = reconstruct_surface_direct_integration(Gx, Gy, pixelsize)
            reconstruction_type = "DirectIntegration"
        else:
            log(logFile,"Error, unknown reconstruction mode")
            logFile.close()
            return None      

        z *= 1e6 # convert to micrometers
        n,m = z.shape
        X,Y = np.meshgrid(np.arange(0, m), np.arange(0, n))

        if RemoveCurvature:
            # Remove curvature defect
            initial_params = [n/5., m/5., 0]
            initial_params = [100, 100, 0]

            # Minimize the squared difference between z(x, y) and p(x, y)
            result = minimize(objective_function, initial_params, args=(X, Y, z), bounds=[(0.1, None), (0.1, None), (-np.inf, np.inf)])

            # Subtract the parabolic surface
            a, b, c = result.x
            log(logFile,"Curvature defect removal, parameters: Rx = " + str(a) + ", Ry = " + str(b))
            P = parabolic_surface(result.x, X, Y)
            z -= P


        # Plot the surface        
        fig, ax = plt.subplots(figsize=(8,10))
        # img = ax.imshow(z, extent=[0, 1e6*X.shape[1]*pixelsize, 0, 1e6*X.shape[0]*pixelsize], interpolation="none")
        # Rotate the image by 90 degrees
        z = np.rot90(z)
        img = ax.imshow(z, extent=[1e6*X.shape[0]*pixelsize, 0, 0, 1e6*X.shape[1]*pixelsize], interpolation="none")
        ax.set_xlabel(r"$y$, $\mu$m")
        ax.set_ylabel(r"$x$, $\mu$m")
        ax.set_title("Surface, optimized")

        # Create a new set of axes above the main axes
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.5)

        # Add the colorbar to the new set of axes
        cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
        cbar.set_label(r'$z$, $\mu$m')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')

        filename = "Surface_"+reconstruction_type+"_" + timeStamp + ".png"
        plt.tight_layout()
        # Crop figure to remove white space
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        returnImgName = filename
        log(logFile, "Surface reconstructed using " + reconstruction_type + " is saved to " + filename)

        # Plot the surface again in grayscale
        fig,ax = plt.subplots()
        ax.imshow(z, cmap='gray')
        ax.axis('off')
        ax.set_aspect('auto')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        filename = "Surface_BW_" + reconstruction_type + "_" + timeStamp + ".png"
        fig.savefig(filename, dpi=300)


        

        x = np.linspace(0, z.shape[1]*pixelsize, num = z.shape[1])
        y = np.linspace(0, z.shape[0]*pixelsize, num = z.shape[0])
        X, Y = np.meshgrid(x, y)
        # save in ASCII file readable by gnuplot with the format # x, y, z and empty lines between each row (if needed)
        if save_file_type == "CSV":
            filename = "Surface_" + timeStamp + ".csv"
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
            filename = "Surface_" + timeStamp + ".npz"
            np.savez(filename, X=X, Y=Y, Z=z)
            log(logFile,"Surface saved to " + filename)
        # Save as vts (vtk structured grid) file
        elif save_file_type == "VTK":
            filename = "Surface_" + timeStamp + ".vts"
            write_vtk(filename, X, Y, z)
            log(logFile,"Surface saved to " + filename)
        else:
            log(logFile,"Surface is not saved")

        now = datetime.datetime.now()
        log(logFile,"Successfully finished at " + now.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        logFile.close()

        return returnImgName

