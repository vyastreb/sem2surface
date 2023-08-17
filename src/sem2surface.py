#---------------------------------------------------------------------------#
#                                                                           #
# SEM/BSE 3D surface reconstruction: 2. Backend part                        #
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import radon

# Function that outputs log information both in terminal and log file
def log(logFile, text):
    print("*     "+text)
    logFile.write(text + "\n")

def constructSurface(imgNames, pixelsize, Plot_images_decomposition, GaussFilter, sigma):
        # Create a log file with time stamp
        import datetime
        now = datetime.datetime.now()
        logFileName = "log_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        logFile = open(logFileName, "w")
        log(logFile,"All information is saved to " + logFileName)

        # Save the names of the images to the log file
        log(logFile,"Image names:")
        for imgName in imgNames:
            log(logFile,imgName)
        log(logFile,"Pixelsize = " + str(pixelsize))
        

        tmp = plt.imread(imgNames[0])
        # Detect first white line in the image
        cutY = 0
        for i in range(1,tmp.shape[0]):
            if abs(np.mean(tmp[i,:]) - 1437.) < 2:
                cutY = i
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
        # FIXME: take three first images
        U, S, V = np.linalg.svd(CorrMatrix)

        img1 = U[0,0]*imgs[0] + U[1,0]*imgs[1] + U[2,0]*imgs[2]
        img2 = U[0,1]*imgs[0] + U[1,1]*imgs[1] + U[2,1]*imgs[2]
        img3 = U[0,2]*imgs[0] + U[1,2]*imgs[1] + U[2,2]*imgs[2]

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
            plt.title("Image A")

            plt.subplot(2,3,2)
            plt.imshow(imgs[1])
            plt.title("Image B")

            plt.subplot(2,3,3)
            plt.imshow(imgs[2])
            plt.title("Image C")

            plt.subplot(2,3,4)
            plt.imshow(img1)
            plt.title("Image 1")

            plt.subplot(2,3,5)
            plt.imshow(img2)
            plt.title("Image 2")

            plt.subplot(2,3,6)
            plt.imshow(img3)
            plt.title("Image 3")

            plt.tight_layout()
            # Save with a unique identifier with a time tag and save in info in log file
            filename = "Images_decomposition_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".png"
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
        # Put all G1 and G2 values zero outside the circle
        G1c = np.zeros(G1.shape)
        # G2c = np.zeros(G2.shape)
        for i in range(G1.shape[0]):
            for j in range(G1.shape[1]):
                if (i - G1.shape[0]/2)**2 + (j - G1.shape[1]/2)**2 < (G1.shape[0]/2)**2:
                    G1c[i,j] = G1[i,j]
                    # G2c[i,j] = G2[i,j]
                    

        R1 = radon(G1c, theta=theta, circle=True)
        # R2 = radon(G2c, theta=theta, circle=True)

        # Find the angle of the unknown direction
        fig,ax = plt.subplots()
        plt.xlim([0, 180])
        plt.xlabel("Projection angle (deg)")
        plt.ylabel("Sum of Radon transform")

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
        filename = "RadonTransform_sum_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".pdf"
        fig.savefig(filename)
        log(logFile,"Radon transform sum saved to " + filename)

        # Compute gradients along identified directions
        Gx = np.cos(theta1*np.pi/180.)*G1 + np.cos(theta2*np.pi/180.)*G2
        Gy = np.sin(theta1*np.pi/180.)*G1 + np.sin(theta2*np.pi/180.)*G2

        # Remove macroscopic tilt
        Gx -= np.mean(Gx)
        Gy -= np.mean(Gy)

        if Plot_images_decomposition:
            # Plot the gradients
            fig,ax = plt.subplots(1,2,figsize=(15,8))
            plt.subplot(1,2,1)
            bar = plt.imshow(Gx)
            plt.colorbar(bar)
            plt.title("Gx")

            plt.subplot(1,2,2)
            bar = plt.imshow(Gy)
            plt.colorbar(bar)
            plt.title("Gy")

            plt.tight_layout()
            # Save with a unique identifier with a time tag and save in info in log file
            filename = "Gradients_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".png"
            fig.savefig(filename, dpi=300)
            log(logFile,"Gradients along principal directions are saved to " + filename)

        # ##############################################################################
        # # Use Frankot and Chellappa method to integrate surface from two gradients
        # # See: R.T. Frankot, R. Chellappa (1988). Method for enforcing integrability in shape from shading algorithms, IEEE Trans. Pattern Anal. Mach. Intell. 10(4):439-451.
        # ##############################################################################

        # def reconstruct_surface_FFT(Gx, Gy):
        #     # Fourier transform of the gradients
        #     Cx = np.fft.fft2(Gx)
        #     Cy = np.fft.fft2(Gy)

        #     # Image size
        #     n, m = Gx.shape

        #     # Wave numbers
        #     kx = np.fft.fftshift(np.arange(0, m) - m / 2) * (2 * np.pi / m)
        #     ky = np.fft.fftshift(np.arange(0, n) - n / 2) * (2 * np.pi / n)

        #     # Expand to 2D matrices
        #     Kx, Ky = np.meshgrid(kx, ky)

        #     # Integrate in Fourier space
        #     C = -1j * (Kx * Cx + Ky * Cy) / (Kx**2 + Ky**2)

        #     # Don't divide by zero (set DC to zero)
        #     C[0, 0] = 0

        #     # Return to real space
        #     z = np.real(np.fft.ifft2(C))

        #     # Set the mean value to zero
        #     z = z - np.mean(z)

        #     return z

        # z = reconstruct_surface_FFT(Gx, Gy).real
        # # Remove mean
        # z -= np.mean(z, axis=0)
        # # z -= np.mean(z, axis=1)[:, np.newaxis]

        # # Plot the surface
        # fig,ax = plt.subplots()
        # bar = plt.imshow(z)
        # plt.colorbar(bar)
        # plt.title("Surface")
        # # Save with a unique identifier with a time tag and save in info in log file
        # filename = "Surface_FFT_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".png"
        # fig.savefig(filename, dpi=300)
        # log(logFile,"Surface reconstructed using FFT is saved to " + filename + "\n")

        
        # Direct integration
        def integrate_profiles(Gx, Gy):            
            # Integrate along x-direction
            int_x = np.cumsum(Gx, axis=0)
            int_x_aligned = np.zeros(int_x.shape)
            int_x_aligned[0,:] = int_x[0,:]
            for j in range(1, int_x.shape[1]):
                int_x_aligned[:,j] = int_x[:,j] + np.mean(int_x_aligned[:,j-1] - int_x[:,j])
            
            # Integrate along y-direction
            int_y = np.cumsum(Gy, axis=1)
            int_y_aligned = np.zeros(int_y.shape)
            int_y_aligned[:,0] = int_y[:,0]
            for i in range(1, int_y.shape[0]):
                int_y_aligned[i,:] = int_y[i,:] + np.mean(int_y_aligned[i-1,:] - int_y[i,:])
            
            # Assemble the surface by averaging
            z = 0.5 * (int_x_aligned + int_y_aligned)
            
            # Adjust the mean value to zero
            z = z - np.mean(z)
            
            return z

        z = integrate_profiles(Gx, Gy)
        z -= np.mean(z, axis=0)
        # z -= np.mean(z, axis=1)[:, np.newaxis]

        # Plot the surface
        fig,ax = plt.subplots()
        bar = plt.imshow(z)
        plt.colorbar(bar)
        plt.title("Surface")
        # Save with a unique identifier with a time tag and save in info in log file
        filename = "Surface_DirectIntegration_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".png"
        fig.savefig(filename, dpi=300)
        log(logFile,"Surface reconstructed using direct integration is saved to " + filename)


        # Plot the surface again
        fig,ax = plt.subplots()
        ax.imshow(z, cmap='gray')

        # Turn off the axis
        ax.axis('off')
        ax.set_aspect('auto')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        # Save with a unique identifier with a time tag and save in info in log file
        imgName = "SurfaceF_DirectIntegration_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".png"
        fig.savefig(imgName, dpi=300)

        # save in ASCII file readable by gnuplot with the format # x, y, z and empty lines between each row
        x = np.linspace(0, z.shape[0]*pixelsize, num=z.shape[1])
        y = np.linspace(0, z.shape[1]*pixelsize, num = z.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = z
        # Save the surface in a file and save info in log file
        filename = "Surface_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".dat"
        log(logFile,"Surface saved to " + filename)
        with open(filename, "w") as f:
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    f.write("{0:.6f} {1:.6f} {2:.6f}\n".format(X[i,j], Y[i,j], Z[i,j]))
                f.write("\n")
        # Save as npz file
        filename = "Surface_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".npz"
        np.savez(filename, X=X, Y=Y, Z=Z)
        log(logFile,"Surface saved to " + filename)
        log(logFile,"Successfully finished at " + now.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        logFile.close()

        return imgName

