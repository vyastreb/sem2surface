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
# Get the absolute path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(src_path)
import sem2surface as s2s

# Load the data
current_dir = os.path.dirname(os.path.abspath(__file__))
imgNames = [os.path.join(current_dir, 'Indent_20230912_7kV_spot4.5_FOV100um_multiBSE_A_05.tif'),    
            os.path.join(current_dir, 'Indent_20230912_7kV_spot4.5_FOV100um_multiBSE_C_05.tif'),
            os.path.join(current_dir, 'Indent_20230912_7kV_spot4.5_FOV100um_multiBSE_B_05.tif')]

GaussFilter = False
sigma = 1.
ReconstructionMode = "FFT"
cutoff_frequency = 0.0
RemoveCurvature = True
Time_stamp = False
Plot_images_decomposition = True
ZscalingFactorPerPixel = 2.1727243e+02

pixelsize = s2s.get_pixel_width(imgNames[0])
_,_,_,_,message = s2s.constructSurface(imgNames, 
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
                         logFile=None)
if message == "":
    print("Successfully reconstructed the surface")
else:
    print("Error: ", message)

