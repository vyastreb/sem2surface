#---------------------------------------------------------------------------#
#                                                                           #
# SEM/BSE 3D surface reconstruction: main file to be used without GUI       #
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

import os
import sys
# Get the path to the src folder
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(src_path)
import sem2surface as s2s

# Load the data
current_dir = os.path.dirname(os.path.abspath(__file__))
imgNames = [os.path.join('Indent_20230912_7kV_spot4.5_FOV100um_multiBSE_A_05.tif'),
            os.path.join('Indent_20230912_7kV_spot4.5_FOV100um_multiBSE_C_05.tif'),
            os.path.join('Indent_20230912_7kV_spot4.5_FOV100um_multiBSE_B_05.tif')]

GaussFilter = False
sigma = 1.
ReconstructionMode = "FFT"
cutoff_frequency = 0.0
RemoveCurvature = True
Time_stamp = False
Plot_images_decomposition = True
ZscalingFactorPerPixel = 2.1727243e+02
Z_ref = 26          # For steel (main element Fe, Z=26)
Z_current = 26      # The same as the reference material

"""
    # Example of how the difference in tilt sensitivity between reference material and current material can be taken into account
    # Reference material on which the indentation was carried out (steel, i.e. mainly Fe (Z=26))
    # Current material of indentation (CrN, i.e. Cr (Z=24) and N (Z=7) in proportions 1:1, in mass percentage, w_Cr=24/31, w_N=7/31
    # so the effetive Z_CrN = 24*24/31+7*7/31=20.16)
    Z_CrN = 20.16
    # Backscattering coefficient models from [1] Böngeler, R., Golla, U., Kässens, M., Reimer, L., Schindler, B., Senkel, R. and Spranck, M., 1993. Electron‐specimen interactions in low‐voltage scanning electron microscopy. Scanning, 15(1), pp.1-18. DOI: https://doi.org/10.1002/sca.4950150102
    # See function backscattering_coefficient_2 in sem2surface.py for details
    def backscattering_coefficient(phi, Z):
        return (1+np.cos(phi*np.pi/180))**(-9/np.sqrt(Z))
    def backscattering_coefficient_2(phi, Z):
        return 0.89*(backscattering_coefficient(0, Z)/0.89)**np.cos(phi*np.pi/180)
    # Approximate difference in tilt sensitivity between reference material and current material in the interval 0-20 degrees
    angles = np.linspace(0,20,100)
    tilt_sensitivity_factor = np.mean(s2s.backscattering_coefficient_2(angles, Z_ref) / s2s.backscattering_coefficient_2(angles, Z_CrN))
"""

pixelsize = s2s.get_pixel_width(imgNames[0])
imgName,X,Y,Z,message = s2s.constructSurface(imgNames, 
                         Plot_images_decomposition, 
                         GaussFilter, 
                         sigma, 
                         ReconstructionMode,
                         RemoveCurvature, 
                         cutoff_frequency=cutoff_frequency, 
                         save_file_type="VTK", 
                         time_stamp=Time_stamp, 
                         pixelsize=pixelsize, 
                         ZscalingFactorPerPixel=ZscalingFactorPerPixel,
                         Z_ref=Z_ref,
                         Z_current=Z_current,
                         logFile=None)
if message == "":
    print("Successfully reconstructed the surface")
else:
    print("Error: ", message)

