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

import sem2surface as s2s

# TODO: add scaling factor

# Load the data
imgNames = ["P33_scale2s1_detectorA.tif",\
            "P33_scale2s1_detectorB.tif",\
            "P33_scale2s1_detectorC.tif"]

imgNames = ["P33_scale1_detectorA.tif",\
            "P33_scale1_detectorB.tif",\
            "P33_scale1_detectorC.tif"]

Plot_images_decomposition = False
GaussFilter = False
sigma = 1.
ReconstructionMode = "FFT" # "FFT" or "DirectIntegration"
cutoff_frequency = 0.0
RemoveCurvature = True
Time_stamp = False
Plot_images_decomposition = True

pixelsize = s2s.get_pixel_width(imgNames[0])
_ = s2s.constructSurface(imgNames, 
                         Plot_images_decomposition, 
                         GaussFilter, 
                         sigma, 
                         ReconstructionMode,
                         RemoveCurvature, 
                         cutoff_frequency=cutoff_frequency, 
                         save_file_type="VTK", 
                         time_stamp=Time_stamp, 
                         pixelsize=pixelsize, 
                         logFile=None)


