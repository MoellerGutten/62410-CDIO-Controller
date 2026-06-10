import os
from pathlib import Path

# --- Configuration ---
# Update these paths if you are fixing the 'train' or 'val' folder
IMAGES_DIR = Path("image_recon/YOLO_data_6.0/images/train")
LABELS_DIR = Path("image_recon/YOLO_data_6.0/labels/train")

# Assuming your images are .jpg. Change to ".png" if necessary.
IMAGE_EXT = ".jpg" 
LABEL_EXT = ".txt"

def fix_and_rename_dataset():
    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        print("Error: Could not find the specified directories. Please check your paths.")
        return

    # 1. Grab all files without their extensions
    image_files = {f.stem: f for f in IMAGES_DIR.glob(f"*{IMAGE_EXT}")}
    label_files = {f.stem: f for f in LABELS_DIR.glob(f"*{LABEL_EXT}")}

    # 2. Find the mismatch
    images_only = set(image_files.keys()) - set(label_files.keys())
    labels_only = set(label_files.keys()) - set(image_files.keys())

    print("--- Mismatch Report ---")
    if images_only:
        print(f"⚠️ Found {len(images_only)} extra image(s) with no label: {images_only}")
    if labels_only:
        print(f"⚠️ Found {len(labels_only)} extra label(s) with no image: {labels_only}")
    if not images_only and not labels_only:
        print("Folders are already perfectly matched!")

    # 3. Get only the matching pairs
    matching_stems = sorted(list(set(image_files.keys()) & set(label_files.keys())))
    print(f"\nFound {len(matching_stems)} perfect pairs. Starting renaming process...")

    # 4. Safe Renaming (Two-step process)
    # We rename to a temporary name first. If we don't do this, renaming a file 
    # to "1.jpg" might accidentally overwrite an existing file originally named "1.jpg".
    
    temp_image_paths = []
    temp_label_paths = []

    # Step 4a: Rename to temporary names
    for i, stem in enumerate(matching_stems, start=1):
        old_img = image_files[stem]
        old_lbl = label_files[stem]

        temp_img = IMAGES_DIR / f"temp_rename_{i}{IMAGE_EXT}"
        temp_lbl = LABELS_DIR / f"temp_rename_{i}{LABEL_EXT}"

        old_img.rename(temp_img)
        old_lbl.rename(temp_lbl)

        # Store the target final name
        temp_image_paths.append((temp_img, f"{i}{IMAGE_EXT}"))
        temp_label_paths.append((temp_lbl, f"{i}{LABEL_EXT}"))

    # Step 4b: Rename from temporary names to final "1-507" names
    for temp_img, final_name in temp_image_paths:
        temp_img.rename(temp_img.parent / final_name)

    for temp_lbl, final_name in temp_label_paths:
        temp_lbl.rename(temp_lbl.parent / final_name)

    print(f"\n✅ Success! Renamed {len(matching_stems)} pairs from 1 to {len(matching_stems)}.")
    
    # Optional tip for the leftover file
    if images_only or labels_only:
        print("\nNote: The extra mismatched files were left untouched with their original names.")

if __name__ == "__main__":
    fix_and_rename_dataset()