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
# V.A. Yastrebov, CNRS, MINES Paris, Aug, 2023                              #
# Licence: BSD 3-Clause                                                     #
#                                                                           #
# Code constructed using GPT4 with CoderPad plugin and Copilot in VSCode    #
#                                                                           #
#---------------------------------------------------------------------------# 

import numpy as np
import sem2surface as s2s

# Load the data
imgNames = ["00_P33_7kV_spot4.5_FOV500um_multiABS_A.tif", "00_P33_7kV_spot4.5_FOV500um_multiABS_B.tif", "00_P33_7kV_spot4.5_FOV500um_multiABS_C.tif"]

Plot_images_decomposition = False
GaussFilter = False
sigma = 1.
ReconstructionMode = "FFT" # "FFT" or "DirectIntegration"  # FIXME bring it to the GUI

_ = s2s.constructSurface(imgNames, Plot_images_decomposition, GaussFilter, sigma, ReconstructionMode)
