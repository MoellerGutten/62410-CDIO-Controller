import os
import glob
import math

def is_duplicate(box1, box2, threshold=0.02):
    if box1[0] != box2[0]: return False
    dist = math.sqrt((box1[1] - box2[1])**2 + (box1[2] - box2[2])**2)
    return dist < threshold

def master_merge(main_folder, extra_folder, new_robot_id, dup_threshold=0.02):
    padding = " " + " ".join(["0.000000"] * 12)
    
    for split in ['train', 'val']:
        main_split_dir = os.path.join(main_folder, 'labels', split)
        extra_split_dir = os.path.join(extra_folder, 'labels', split)
        
        if not os.path.exists(main_split_dir):
            continue
            
        # Get a list of all filenames from BOTH folders to ensure we don't miss anything
        main_files = [os.path.basename(f) for f in glob.glob(os.path.join(main_split_dir, '*.txt'))]
        extra_files = [os.path.basename(f) for f in glob.glob(os.path.join(extra_split_dir, '*.txt'))] if os.path.exists(extra_split_dir) else []
        all_filenames = set(main_files + extra_files)
        
        print(f"Processing {len(all_filenames)} files in '{split}' folder...")
        
        for filename in all_filenames:
            if filename == "classes.txt": continue
            
            final_lines = []
            accepted_boxes = []
            
            main_filepath = os.path.join(main_split_dir, filename)
            extra_filepath = os.path.join(extra_split_dir, filename)
            
            # --- 1. PROCESS THE BALLS/CROSS (Main Folder) ---
            if os.path.exists(main_filepath):
                with open(main_filepath, 'r') as f:
                    for line in f.readlines():
                        if not line.strip(): continue
                        parts = line.strip().split()
                        
                        box_data = None
                        
                        # If it's already a 5-number Bounding Box
                        if len(parts) == 5:
                            box_data = [parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                            
                        # If it's a Polygon (>5 numbers), convert it to a Box!
                        elif len(parts) > 5 and len(parts) != 17:
                            class_id = parts[0]
                            coords = [float(x) for x in parts[1:]]
                            x_coords = coords[0::2]
                            y_coords = coords[1::2]
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            
                            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
                            width, height = x_max - x_min, y_max - y_min
                            box_data = [class_id, center_x, center_y, width, height]
                            
                        # If it somehow already successfully converted to a 17-number pose line
                        elif len(parts) == 17:
                            final_lines.append(line.strip())
                            continue
                            
                        if box_data:
                            # Check if it's a duplicate
                            is_dup = any(is_duplicate(box_data, acc, dup_threshold) for acc in accepted_boxes)
                            
                            if not is_dup:
                                accepted_boxes.append(box_data)
                                # Format nicely and stick the 12 zeros on the end
                                formatted_box = f"{box_data[0]} {box_data[1]:.6f} {box_data[2]:.6f} {box_data[3]:.6f} {box_data[4]:.6f}"
                                final_lines.append(formatted_box + padding)
                                
            # --- 2. PROCESS THE ROBOT (Extra Folder) ---
            if os.path.exists(extra_filepath):
                with open(extra_filepath, 'r') as f:
                    for line in f.readlines():
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) > 0:
                                # Change the Class ID to the new Robot number
                                parts[0] = str(new_robot_id)
                                final_lines.append(" ".join(parts))
                                
            # --- 3. SAVE THE COMBINED FILE ---
            if final_lines:
                os.makedirs(main_split_dir, exist_ok=True) # Just in case
                with open(main_filepath, 'w') as f:
                    f.write('\n'.join(final_lines))

if __name__ == "__main__":
    MAIN_FOLDER = "image_recon/YOLO_data_4.0"      
    EXTRA_FOLDER = "image_recon/yolo_data_extra"   
    ROBOT_NEW_ID = 3  # Change to your Robot ID in data.yaml
    
    print("Running the Master Dataset Merge...")
    master_merge(MAIN_FOLDER, EXTRA_FOLDER, ROBOT_NEW_ID)
    print("Done! Check your YOLO_data_4.0 folder.")