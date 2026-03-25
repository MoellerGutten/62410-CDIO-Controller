import os
import shutil
import zipfile

# --- Verify these paths match your computer ---
IMAGES_DIR = "data/data_2.0/rest_of_images_2.0" 
LABELS_DIR = "runs/segment/predict/labels" 
# ----------------------------------------------

ZIP_NAME = "PERFECT_CVAT_UPLOAD.zip"

def main():
    print("Building CVAT-friendly zip file...")
    
    # 1. Create a temporary folder structure
    temp_dir = "temp_cvat_folder"
    images_train = os.path.join(temp_dir, "images", "train")
    labels_train = os.path.join(temp_dir, "labels", "train")
    
    os.makedirs(images_train, exist_ok=True)
    os.makedirs(labels_train, exist_ok=True)
    
    # 2. Copy images and labels, and build train.txt
    train_txt_lines = []
    count = 0
    
    for filename in os.listdir(IMAGES_DIR):
        if not filename.lower().endswith(".jpg"): 
            continue
            
        base_name = os.path.splitext(filename)[0]
        txt_name = base_name + ".txt"
        
        src_img = os.path.join(IMAGES_DIR, filename)
        src_txt = os.path.join(LABELS_DIR, txt_name)
        
        # Only process if both the image and the AI label exist
        if os.path.exists(src_txt):
            shutil.copy(src_img, os.path.join(images_train, filename))
            shutil.copy(src_txt, os.path.join(labels_train, txt_name))
            
            # Add to train.txt mapping
            train_txt_lines.append(f"images/train/{filename}\n")
            count += 1
            
    # 3. Create the crucial train.txt file
    with open(os.path.join(temp_dir, "train.txt"), "w") as f:
        f.writelines(train_txt_lines)
        
    # 4. Create the perfect data.yaml
    yaml_content = """path: ./
train: train.txt
val: train.txt
nc: 6
names: ['WBall', 'Border', 'X', 'GoalA', 'GoalB', 'OBall']
"""
    with open(os.path.join(temp_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)
        
    # 5. Zip it perfectly (no outer folders!)
    print(f"Zipping {count} matched images and labels...")
    with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Keep paths relative inside the zip
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
                
    # Cleanup temp folder
    shutil.rmtree(temp_dir)
    print(f"Done! Created '{ZIP_NAME}'.")

if __name__ == "__main__":
    main()