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
imgNames = [os.path.join(current_dir, 'P33_scale1_detectorA.tif'),
            os.path.join(current_dir, 'P33_scale1_detectorB.tif'),
            os.path.join(current_dir, 'P33_scale1_detectorC.tif')]

GaussFilter = False
sigma = 1.
ReconstructionMode = "FFT"
cutoff_frequency = 0.0
RemoveCurvature = True
Time_stamp = False
Plot_images_decomposition = True
ZscalingFactor = 1.4076e-5 # *10 # if you need to have an enhanced scale for visualization needs add an extra multiplier

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
                         ZscalingFactor=ZscalingFactor,
                         logFile=None)
if message == "":
    print("Successfully reconstructed the surface")
else:
    print("Error: ", message)
