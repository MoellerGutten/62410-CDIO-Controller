import os
from pathlib import Path
import shutil

# --- Configuration ---
root_dir = Path("image_recon")
dataset_old_root = root_dir / "YOLO_data_3.0"
dataset_new_root = root_dir / "YOLO_data_4.0"

# --- Class Mapping ---
# Old: 0:WBall, 1:OBall, 2:GoalA... 7:BottomLeft, 8:X
# New: 0:WBall, 1:OBall, 2:X, 3:Robot, 4:RobotHeading
class_id_map = {
    0: 0,  
    1: 1,  
    8: 2   
    # 2,3,4,5,6,7 are DELETED
}

# --- Core Processing Function ---
def clean_and_merge_split(source_split, target_split="train"):
    """
    Cleans label files from the source_split and moves everything 
    into the specified target_split (defaults to 'train').
    """
    print(f"[Processing] Moving old '{source_split}' into new '{target_split}'...")
    
    # Path setup
    old_img_dir = dataset_old_root / "images" / source_split
    old_lbl_dir = dataset_old_root / "labels" / source_split
    new_img_dir = dataset_new_root / "images" / target_split
    new_lbl_dir = dataset_new_root / "labels" / target_split

    # Create destination directories if they are missing
    new_img_dir.mkdir(parents=True, exist_ok=True)
    new_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not old_img_dir.exists() or not old_lbl_dir.exists():
        print(f"  [Skipping] Source folders for '{source_split}' do not exist.")
        return

    # Create a unique prefix so old train and old val files don't collide
    prefix = f"v3_{source_split}_"

    images_found = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images_found.extend(old_img_dir.glob(ext))
    
    if not images_found:
        print(f"  [Warning] No image files found in {old_img_dir}.")
        return

    num_processed = 0

    for img_path in images_found:
        img_filename_base = img_path.stem
        lbl_filename = img_filename_base + ".txt"
        lbl_path = old_lbl_dir / lbl_filename
        
        # Apply the safe prefix
        merged_img_filename = prefix + img_path.name
        merged_lbl_filename = prefix + lbl_filename
        merged_img_path = new_img_dir / merged_img_filename
        merged_lbl_path = new_lbl_dir / merged_lbl_filename

        # If no label file exists for the image
        if not lbl_path.exists():
            shutil.copy2(img_path, merged_img_path)
            if merged_lbl_path.exists(): os.remove(merged_lbl_path)
            continue

        # Clean annotations
        cleaned_lines = []
        with open(lbl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                parts = line.split()
                if not parts: continue

                try:
                    old_class_id = int(parts[0])
                except ValueError:
                    continue

                if old_class_id in class_id_map:
                    new_class_id = class_id_map[old_class_id]
                    reassigned_line = f"{new_class_id} " + " ".join(parts[1:]) + '\n'
                    cleaned_lines.append(reassigned_line)
        
        # Save processed files
        if not cleaned_lines:
            shutil.copy2(img_path, merged_img_path)
            if merged_lbl_path.exists(): os.remove(merged_lbl_path)
        else:
            shutil.copy2(img_path, merged_img_path)
            with open(merged_lbl_path, 'w') as f:
                f.writelines(cleaned_lines)
            num_processed += 1
    
    print(f"  [Complete] Moved {len(images_found)} images and {num_processed} valid label files to '{target_split}'.")

# --- Execution ---
if __name__ == "__main__":
    print("--- Starting Dataset Merge (Pooling to Train) ---")
    
    # 1. Pool the old 'train' data into the new 'train' folder
    clean_and_merge_split(source_split="train", target_split="train")
    
    # 2. Pool the old 'val' data into the new 'train' folder too!
    clean_and_merge_split(source_split="val", target_split="train")

    print("\n--- Process Complete ---")
    print(f"All cleaned data is now in: {dataset_new_root}/images/train")
    print("You can now safely run your 80/20 split script!")