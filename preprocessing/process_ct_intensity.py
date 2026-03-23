import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse

def process_intensity(input_ct_dir, input_label_dir, output_dir):
    """
    Computes global mean CT intensity per organ and generates new label maps.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize global statistics for CT values and pixel counts
    global_sum = {}   # Sum of intensity values per organ ID
    global_count = {} # Total pixel count per organ ID

    # Get sorted list of CT files
    ct_files = sorted([f for f in os.listdir(input_ct_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    # --- First Pass: Accumulate Global Statistics ---
    for ct_file in tqdm(ct_files, desc="Pass 1/2: Accumulating Statistics"):
        ct_path = os.path.join(input_ct_dir, ct_file)
        label_path = os.path.join(input_label_dir, ct_file)
        
        if not os.path.exists(label_path):
            continue

        # Read images (Unchanged scale, float32 for calculation precision)
        ct = cv2.imread(ct_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Get unique organ IDs (excluding background 0)
        unique_labels = np.unique(label)
        unique_labels = unique_labels[unique_labels > 0]

        for organ_id in unique_labels:
            organ_mask = (label == organ_id)
            
            # Update global accumulators
            global_sum[organ_id] = global_sum.get(organ_id, 0) + np.sum(ct[organ_mask])
            global_count[organ_id] = global_count.get(organ_id, 0) + np.sum(organ_mask)

    # Compute global mean intensity for each organ: mean = sum / count
    global_mean = {oid: (global_sum[oid] / global_count[oid] if global_count[oid] > 0 else 0) 
                   for oid in global_sum}

    # --- Second Pass: Generate Intensity-Based Labels ---
    for ct_file in tqdm(ct_files, desc="Pass 2/2: Generating Labels"):
        label_path = os.path.join(input_label_dir, ct_file)
        if not os.path.exists(label_path):
            continue

        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        label_intensity = np.zeros_like(label, dtype=np.float32)

        unique_labels = np.unique(label)
        unique_labels = unique_labels[unique_labels > 0]

        # Fill each organ region with its corresponding global mean
        for organ_id in unique_labels:
            if organ_id in global_mean:
                label_intensity[label == organ_id] = global_mean[organ_id]

        # Normalize/Clip to [0, 255] for saving as 8-bit image
        # Adjust p_min/p_max based on your specific CT windowing requirements
        p_min, p_max = 0, 255
        label_intensity_clipped = np.clip(label_intensity, p_min, p_max)
        label_intensity_norm = ((label_intensity_clipped - p_min) / (p_max - p_min) * 255).astype(np.uint8)

        # Save result
        output_path = os.path.join(output_dir, ct_file)
        cv2.imwrite(output_path, label_intensity_norm)

    print(f"\nProcessing complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT Organ Intensity Normalization Tool")
    parser.add_argument("--ct_dir", type=str, default="./data/images", help="Path to raw CT images")
    parser.add_argument("--label_dir", type=str, default="./data/labels", help="Path to organ masks")
    parser.add_argument("--output_dir", type=str, default="./data/output", help="Path to save processed labels")
    
    args = parser.parse_args()
    process_intensity(args.ct_dir, args.label_dir, args.output_dir)
