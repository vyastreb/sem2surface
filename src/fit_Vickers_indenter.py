#---------------------------------------------------------------------------#
#                                                                           #
# SEM/BSE 3D surface reconstruction: tool to identify z-scaling factor      #
# based on the Vicker's indenter imprint on the same material (or material  #
# with a close atomic weight)                                               #
#                                                                           #
# V.A. Yastrebov, CNRS, MINES Paris, Aug 2023-Dec 2024                      #
# Licence: BSD 3-Clause                                                     #
#                                                                           #
# Aided by :                                                                #
#  - GPT4 with CoderPad plugin                                              #
#  - Copilot in VSCode                                                      #
#  - Claude 3.5 Sonnet in cursor.                                           #
#---------------------------------------------------------------------------#

import sem2surface as s2s
import numpy as np
import matplotlib.pyplot as plt
 
# Load indentation SEM data, at least 3 images
imgNames = ["Indent_20230912_7kV_spot4.5_FOV100um_multiBSE_A_05.tif",\
            "Indent_20230912_7kV_spot4.5_FOV100um_multiBSE_C_05.tif",\
            "Indent_20230912_7kV_spot4.5_FOV100um_multiBSE_B_05.tif"]
# Reconstruction parameters
GaussFilter = False
sigma = 1.
ReconstructionMode = "FFT" 
cutoff_frequency = 0.0
RemoveCurvature = False # Do not use the default curvature removal, it is adjusted for the Vicker's indenter imprint here.
Time_stamp = False
Plot_images_decomposition = False
ZscalingFactor = 1.e-4
# Indenter parameters to be defined by the user!
R = 6 # (micrometers) Radius inside the indenter, should be big enough to include as much of the indenter as possible, but small enough to be kept entirely inside the indenter imprint region

pixelsize = s2s.get_pixel_width(imgNames[0])
_,X,Y,z,message = s2s.constructSurface(imgNames, 
                         Plot_images_decomposition, 
                         GaussFilter, 
                         sigma, 
                         ReconstructionMode,
                         RemoveCurvature=True, 
                         cutoff_frequency=cutoff_frequency, 
                         save_file_type="VTK", 
                         time_stamp=Time_stamp, 
                         pixelsize=pixelsize, 
                         ZscalingFactor=ZscalingFactor,
                         logFile=None)
if message != "":
    print(message)

#  Vicker's indenter pyramid with 136° angle at opposing faces
#        136°
#      \_----_/
#       \    /
#        \  /
#         \/
#
def VickerIndenter(X,Y,X0,Y0,R,rot,depth,scale):
    angle = 136 * np.pi / 180. # in radians
    Xprime = (X-X0)*np.cos(rot) - (Y-Y0)*np.sin(rot)
    Yprime = (X-X0)*np.sin(rot) + (Y-Y0)*np.cos(rot)
    Z = np.zeros(X.shape)

    # Create masks for each condition
    mask1 = (Xprime >= 0) & (np.abs(Yprime) < Xprime)
    mask2 = (Xprime < 0) & (np.abs(Yprime) < np.abs(Xprime))
    mask3 = (Yprime >= 0) & (np.abs(Xprime) < np.abs(Yprime))
    mask4 = (Yprime < 0) & (np.abs(Xprime) < np.abs(Yprime))

    # Initialize Z with depth
    Z = np.full(X.shape, depth)

    # Apply calculations using masks
    tan_half = np.tan(np.pi/2 - angle/2.)
    Z[mask1] += Xprime[mask1] * tan_half
    Z[mask2] -= Xprime[mask2] * tan_half
    Z[mask3] += Yprime[mask3] * tan_half
    Z[mask4] -= Yprime[mask4] * tan_half

    mask = (X-X0)**2 + (Y-Y0)**2 > R**2
    Z[mask] = np.nan
    Z *= scale
    return Z

# Work in micrometers
pixelsize *= 1e6
sign = 1.
# Find the appropriate scaling for Vicker's imprint
meanz = np.nanmean(z)
if np.nanmax(z) - meanz > meanz - np.nanmin(z):
    z *= -1.
    sign = -1.
# Find the deepest point and assign it to the Vicker's pyramid summit
Y0i,X0i = np.where(np.nanmin(z) == z)

X0 = X0i[0]*pixelsize
Y0 = Y0i[0]*pixelsize
limx0 = int((X0-R)/pixelsize)
limx1 = int((X0+R)/pixelsize)
limy0 = int((Y0-R)/pixelsize)
limy1 = int((Y0+R)/pixelsize)

rot = 0
mask = (X-X0)**2 + (Y-Y0)**2 > R**2
zp = z.copy()
zp[mask] = np.nan

maxDepth = np.nanmin(zp)
depth = maxDepth

# ============================================================================ #
#      Identify the orientation of the imprint and fit the scaling factor      #
# ============================================================================ #

# Try different rotations to find the best match
# Initialize best parameters
min_diff = np.inf
best_rot = 0
best_scale = 1

# Multi-step optimization with decreasing angle increments
angle_steps = [10, 2, 0.5, 0.1, 0.02]  # Degrees
search_range = 45  # Initial search range in degrees

for step in angle_steps:
    # Convert to radians
    step_rad = step * np.pi/180
    range_rad = search_range * np.pi/180
    
    # Create angle array centered around current best rotation
    rot_angles = np.arange(best_rot - range_rad, best_rot + range_rad + step_rad, step_rad)
    
    # Test all angles in current range
    for test_rot in rot_angles:
        # Generate indenter surface with test rotation
        zind_test = VickerIndenter(X, Y, X0, Y0, R, test_rot, depth, 1)
        
        # Remove NaN values and align the surfaces
        mask = ~np.isnan(zind_test) & ~np.isnan(zp)
        if not np.any(mask):
            continue
            
        # Calculate scaling factor as ratio of mean values
        test_scale = np.mean(zp[mask] - maxDepth) / np.mean(zind_test[mask] - np.nanmin(zind_test))
        
        # Apply scaling and calculate difference
        zind_scaled = (zind_test - np.nanmin(zind_test)) * test_scale
        diff = np.nansum((zind_scaled[mask] - zp[mask])**2)
        
        # Update best parameters if better match found
        if diff < min_diff:
            min_diff = diff
            best_rot = test_rot
            best_scale = test_scale
    
    # Reduce search range for next iteration
    search_range = 2 * step

rot = best_rot
scale = best_scale
ZscalingFactor /= scale

print(f"\n===============================\n Scaling factor = {ZscalingFactor:.7e}\n===============================\n")

zind = VickerIndenter(X,Y,X0,Y0,R,rot,depth,1)
zind -= np.nanmin(zind)
zp -= np.nanmin(zp)
# zp *= ZscalingFactor
zp /= scale
vmin = 0.
vmax = np.nanmax(zind)
# Create figure with 3 subplots in one row
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
# Add general title
fig.suptitle(f"Vickers' indenter imprint, scaling factor = {ZscalingFactor:.5e}")

# Plot real indenter surface
im1 = ax1.imshow(zp[limy0:limy1,limx0:limx1], 
                 extent=[limy0*pixelsize, limy1*pixelsize, limx0*pixelsize, limx1*pixelsize],
                 vmin=vmin, vmax=vmax, cmap='rainbow')
ax1.set_title("Real Indenter")
fig.colorbar(im1, ax=ax1, label=r"$z$, $\mu$m")
ax1.set_xlabel(r"$x$, $\mu$m")
ax1.set_ylabel(r"$y$, $\mu$m")

# Plot ideal indenter surface
zind_norm = zind[limy0:limy1,limx0:limx1]-np.nanmin(zind)
im2 = ax2.imshow(zind_norm,
                 extent=[limy0*pixelsize, limy1*pixelsize, limx0*pixelsize, limx1*pixelsize],
                 vmin=vmin, vmax=vmax, cmap="rainbow")
ax2.set_title("Ideal Indenter")
fig.colorbar(im2, ax=ax2, label=r"$z$, $\mu$m")
ax2.set_xlabel(r"$x$, $\mu$m")
ax2.set_ylabel(r"$y$, $\mu$m")

# Plot normalized difference
diff = (zind_norm-zp[limy0:limy1,limx0:limx1])
diff_norm = diff / np.nanmean(zp[limy0:limy1,limx0:limx1])
diff_max = 0.9*np.nanmax(diff_norm)
diff_min = -diff_max
im3 = ax3.imshow(diff_norm,
                 extent=[limy0*pixelsize, limy1*pixelsize, limx0*pixelsize, limx1*pixelsize],
                 vmin=diff_min, vmax=diff_max)
ax3.set_title("Normalized Difference")
cbar = fig.colorbar(im3, ax=ax3, cmap='coolwarm')
im3.set_cmap('coolwarm')
cbar.set_label(r"Difference, $z_p/\langle z_p \rangle$")

plt.tight_layout()
fig.savefig(f"Indenter_comparison_{ZscalingFactor:.4e}.png", dpi=300)

# ============================================================================ #
#   Reconstruct the surface with the identified scaling factor and curvatures  #
#   and check once again the difference between the real and ideal indenters   #
# ============================================================================ #

print("ZscalingFactor = ",ZscalingFactor)
pixelsize = s2s.get_pixel_width(imgNames[0])
_,X,Y,z,message = s2s.constructSurface(imgNames, 
                         Plot_images_decomposition, 
                         GaussFilter, 
                         sigma, 
                         ReconstructionMode,
                         RemoveCurvature = True, 
                         cutoff_frequency=cutoff_frequency, 
                         save_file_type="VTK", 
                         time_stamp=False, 
                         pixelsize=pixelsize, 
                         ZscalingFactor=ZscalingFactor,
                         logFile=None)
if message != "":
    print(message)

