import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define File Paths ---
ct_path = r"Data\data_nifty\1.Training_DICOM_603\AI_JIU_GEN_P1548715\CT\A_8\A_8_02_shenAgioRoutine_20170728165801_6.nii.gz"
voi_path = r"Data\ROI\1.Training_ROI_603\AI_JIU_GEN_P1548715\ROI\A_8.nrrd"

# --- 2. Load Images using SimpleITK ---
# SimpleITK handles orientation differences automatically.
# DO NOT use as_closest_canonical() here.
ct_img = sitk.ReadImage(ct_path, sitk.sitkFloat64)
voi_img = sitk.ReadImage(voi_path)

# --- 3. Resample Segmentation to Match CT Grid ---
# Create a resampler object
resampler = sitk.ResampleImageFilter()

# Set the resampler to use the CT's metadata (spacing, origin, direction)
resampler.SetReferenceImage(ct_img)

# IMPORTANT: Use nearest neighbor interpolation for segmentation masks
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetDefaultPixelValue(0) # Pad with background value

# Execute the resampling
voi_resampled = resampler.Execute(voi_img)

# --- 4. Convert to NumPy Arrays for Visualization ---
# Note: SimpleITK's array axis order is (z, y, x)
ct_data = sitk.GetArrayFromImage(ct_img)
voi_data = sitk.GetArrayFromImage(voi_resampled)

print("CT shape:", ct_data.shape)
print("Resampled VOI shape:", voi_data.shape) # Should now match the CT shape

# --- 5. Visualize a Slice ---
# Pick a slice to visualize (note the z-axis is now the first dimension)
z = ct_data.shape[0] // 2

# Normalize CT for display
ct_slice = ct_data[z, :, :]
# Windowing can often be better than simple normalization for CT
# Example windowing for an abdominal CT
window_center = 40
window_width = 400
min_val = window_center - window_width / 2
max_val = window_center + window_width / 2
ct_slice_display = np.clip(ct_slice, min_val, max_val)

voi_slice = voi_data[z, :, :]

plt.figure(figsize=(8, 8))
plt.imshow(ct_slice_display, cmap='gray')
plt.imshow(np.ma.masked_where(voi_slice == 0, voi_slice), cmap='autumn', alpha=0.5)
plt.title(f"Correctly Aligned CT with VOI Overlay (slice {z})")
plt.axis('off')
plt.show()