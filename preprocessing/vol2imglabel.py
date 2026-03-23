import os
import numpy as np
import nibabel as nib
from PIL import Image

print("Script execution started...")

# Reading and filtering directories
dataset_dir = './Totalsegmentator_dataset/'
subjects = [s for s in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, s))]

print(f"Found {len(subjects)} subject directories.")

# Setting the pixel value range and iterating over each subject
p_min, p_max = -200, 500
for idx, subject in enumerate(subjects[9:], start=10):  # Starting from the 10th subject
    print(f"Processing subject {idx}: {subject}")
    fp = os.path.join(dataset_dir, subject)
    
    ct_path = os.path.join(fp, 'ct.nii.gz')
    if not os.path.isfile(ct_path):
        print(f"CT image file not found: {ct_path}")
        continue
    
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    
    # Truncate and normalize CT data
    ct_data = np.clip(ct_data, p_min, p_max)
    ct_data = (ct_data - p_min) / (p_max - p_min)
    
    mask_path = os.path.join(fp, 'mask.nii')
    if not os.path.isfile(mask_path):
        print(f"Mask file not found: {mask_path}")
        continue
    
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    
    # Adjust the shapes if necessary
    h, w, slices = ct_data.shape
    print(f"CT image dimensions: width={w}, height={h}, slices={slices}")
    
    if w > h:
        pad = (w - h) // 2
        ct_data = np.pad(ct_data, ((pad, pad), (0, 0), (0, 0)), mode='constant')
        mask_data = np.pad(mask_data, ((pad, pad), (0, 0), (0, 0)), mode='constant')
    elif h > w:
        pad = (h - w) // 2
        ct_data = np.pad(ct_data, ((0, 0), (pad, pad), (0, 0)), mode='constant')
        mask_data = np.pad(mask_data, ((0, 0), (pad, pad), (0, 0)), mode='constant')
    
    # Process and save each slice
    for j in range(slices):
        if np.max(mask_data[:, :, j]) == 0:
            continue
        img = np.rot90(ct_data[:, :, j])
        img = Image.fromarray((img * 255).astype(np.uint8))
        img = img.resize((256, 256), Image.LANCZOS)
        img_path = f'./image/{subject}_{j+1}.png'
        img.save(img_path)
        
        label = np.rot90(mask_data[:, :, j])
        label = Image.fromarray((label * 255 / label.max()).astype(np.uint8))
        label = label.resize((256, 256), Image.NEAREST)
        label_path = f'./label/{subject}_{j+1}.png'
        label.save(label_path)
    
    print(f"Finished processing {subject}, saved {slices} images and labels.")

print("Script execution completed.") # generate images
