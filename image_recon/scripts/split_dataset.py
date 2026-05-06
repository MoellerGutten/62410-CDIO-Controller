import os
import random
import shutil

# --- Setup your paths here ---
# Point this to your main dataset folder
BASE_DIR = "image_recon/YOLO_data_5.0"

IMAGES_TRAIN = os.path.join(BASE_DIR, "images", "train")
LABELS_TRAIN = os.path.join(BASE_DIR, "labels", "train")

IMAGES_VAL = os.path.join(BASE_DIR, "images", "val")
LABELS_VAL = os.path.join(BASE_DIR, "labels", "val")

# Set your split ratio (0.20 = 20% goes to val, 80% stays in train)
VAL_RATIO = 0.20 

def main():
    print("Starting dataset split...")

    # 1. Create the val folders if they don't exist yet
    os.makedirs(IMAGES_VAL, exist_ok=True)
    os.makedirs(LABELS_VAL, exist_ok=True)

    # 2. Get all images currently sitting in the train folder
    all_images = [f for f in os.listdir(IMAGES_TRAIN) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not all_images:
        print("No images found in the train folder! Check your paths.")
        return

    # 3. Shuffle them randomly so the validation set has a good, unbiased mix
    random.shuffle(all_images)

    # 4. Calculate exactly how many images to move
    val_count = int(len(all_images) * VAL_RATIO)
    images_to_move = all_images[:val_count]

    print(f"Found {len(all_images)} total images in 'train'.")
    print(f"Moving {val_count} images and their labels to 'val' ({VAL_RATIO * 100}% split)...")

    # 5. Move the files!
    moved_count = 0
    for filename in images_to_move:
        # Image paths
        src_img = os.path.join(IMAGES_TRAIN, filename)
        dst_img = os.path.join(IMAGES_VAL, filename)

        # Label paths (Swap the .jpg extension for .txt)
        base_name = os.path.splitext(filename)[0]
        txt_name = base_name + ".txt"
        src_txt = os.path.join(LABELS_TRAIN, txt_name)
        dst_txt = os.path.join(LABELS_VAL, txt_name)

        # Move the image
        shutil.move(src_img, dst_img)

        # Move the corresponding label (if it exists)
        if os.path.exists(src_txt):
            shutil.move(src_txt, dst_txt)
        else:
            print(f"Warning: Label file {txt_name} not found for image {filename}")
            
        moved_count += 1

    print("-" * 30)
    print("Success! Set up complete.")
    print(f"Train folder now has: {len(all_images) - moved_count} images.")
    print(f"Val folder now has: {moved_count} images.")

if __name__ == "__main__":
    main()