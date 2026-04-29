import os
import shutil
from pathlib import Path

def create_clean_dataset(src_dir="image_recon/YOLO_data_4.0", dest_dir="YOLO_data_5.0", class_to_remove="3"):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)

    # 1. Safety Checks
    if not src_path.exists():
        print(f"❌ Error: Could not find source directory '{src_dir}'")
        return

    if dest_path.exists():
        print(f"❌ Error: Destination '{dest_dir}' already exists.")
        print("Please delete or rename it before running this script again.")
        return

    # 2. Copy the entire dataset
    print(f"📁 Copying dataset from '{src_dir}' to '{dest_dir}'... (This might take a moment depending on image count)")
    shutil.copytree(src_path, dest_path)

    # 3. Clean the labels in the NEW directory
    labels_path = dest_path / "labels"
    total_files_modified = 0
    total_lines_removed = 0

    if labels_path.exists():
        print(f"🧹 Scanning '{dest_dir}/labels' to remove class ID '{class_to_remove}'...")
        
        # rglob searches recursively, hitting train/ and val/ folders automatically
        for txt_file in labels_path.rglob("*.txt"):
            with open(txt_file, "r") as file:
                lines = file.readlines()

            filtered_lines = []
            lines_removed_in_file = 0

            for line in lines:
                if line.strip().startswith(f"{class_to_remove} "):
                    lines_removed_in_file += 1
                else:
                    filtered_lines.append(line)

            # Only overwrite if we changed something
            if lines_removed_in_file > 0:
                with open(txt_file, "w") as file:
                    file.writelines(filtered_lines)
                
                total_files_modified += 1
                total_lines_removed += lines_removed_in_file
    else:
        print(f"⚠️ Warning: No 'labels' folder found inside {dest_dir}.")

    # 4. Update the path inside data.yaml
    yaml_path = dest_path / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as file:
            yaml_content = file.read()
        
        # Swaps 'YOLO_data_4.0' for 'YOLO_data_5.0' in the path variable
        yaml_content = yaml_content.replace(src_dir, dest_dir)
        
        with open(yaml_path, "w") as file:
            file.write(yaml_content)
        print(f"📝 Updated data.yaml to point to {dest_dir}.")

    # 5. Summary
    print("\n✅ --- New Dataset Created Successfully ---")
    print(f"Target location:       {dest_dir}/")
    print(f"Label files modified:  {total_files_modified}")
    print(f"Robot labels removed:  {total_lines_removed}")

if __name__ == "__main__":
    create_clean_dataset()