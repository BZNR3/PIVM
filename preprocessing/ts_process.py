import os
import nibabel as nib
import numpy as np
import glob

def main():
    print("Starting processing...")

    # Set dataset path
    dataset_path = '../Totalsegmentator_dataset_v201/'
    subject_paths = [path for path in glob.glob(os.path.join(dataset_path, '*')) if os.path.isdir(path)]

    if not subject_paths:
        print("No subject directories found, please check the path.")
        return

    for subject_path in subject_paths:
        print(f"Processing: {subject_path}")
        
        ct_path = os.path.join(subject_path, 'ct.nii.gz')
        if not os.path.exists(ct_path):
            print(f"CT image not found: {ct_path}")
            continue
        
        ct = nib.load(ct_path).get_fdata()
        
        # Initialize the mask as a zero matrix
        mask = np.zeros_like(ct)
        found_segmentation = False

        segmentation_paths = {
            'lung_upper_lobe_left': 'lung_upper_lobe_left.nii.gz',
            'lung_lower_lobe_left': 'lung_lower_lobe_left.nii.gz',
            'lung_upper_lobe_right': 'lung_upper_lobe_right.nii.gz',
            'lung_middle_lobe_right': 'lung_middle_lobe_right.nii.gz',
            'lung_lower_lobe_right': 'lung_lower_lobe_right.nii.gz',
            'adrenal_gland_right': 'adrenal_gland_right.nii.gz',
            'adrenal_gland_left': 'adrenal_gland_left.nii.gz',
            'spleen': 'spleen.nii.gz',
            'kidney_right': 'kidney_right.nii.gz',
            'kidney_left': 'kidney_left.nii.gz',
            'gallbladder': 'gallbladder.nii.gz',
            'liver': 'liver.nii.gz',
            'stomach': 'stomach.nii.gz',
            'pancreas': 'pancreas.nii.gz'
        }
        
        for organ, filename in segmentation_paths.items():
            seg_path = os.path.join(subject_path, 'segmentations', filename)
            if not os.path.exists(seg_path):
                print(f"Segmentation image not found: {seg_path}")
                continue
            found_segmentation = True
            seg = nib.load(seg_path).get_fdata()
            mask += seg * (list(segmentation_paths.keys()).index(organ) + 1)
        
        if not found_segmentation:
            print("No segmentation images found, skipping this subject.")
            continue

        # Save the mask
        mask_nifti = nib.Nifti1Image(mask.astype(np.int16), affine=np.eye(4))
        save_path = os.path.join(subject_path, 'mask.nii')
        nib.save(mask_nifti, save_path)
        print(f"Mask saved to: {save_path}")

if __name__ == "__main__":
    main() # generate labels
