import os
import json
import cv2

# --- Your exact paths ---
IMAGES_DIR = "image_recon/data/data_2.0/rest_of_photos_2.0" 
LABELS_DIR = "runs/segment/predict/labels" 

# Matches your data.yaml perfectly
CLASSES = ['WBall', 'Border', 'X', 'GoalA', 'GoalB', 'OBall']

def main():
    print("Converting YOLO text files to a single COCO JSON file...")
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)]
    }

    ann_id = 1
    image_id = 1

    for filename in os.listdir(IMAGES_DIR):
        if not filename.lower().endswith(".jpg"): 
            continue

        # 1. Read the image to get its exact width and height
        img_path = os.path.join(IMAGES_DIR, filename)
        img = cv2.imread(img_path)
        if img is None: 
            continue
            
        height, width = img.shape[:2]

        # 2. Add image info to COCO
        coco_data["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        # 3. Look for the matching AI text file
        txt_name = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(LABELS_DIR, txt_name)

        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 7: 
                    continue # Ignore invalid polygons
                    
                cat_id = int(parts[0])

                # Convert YOLO normalized math (0.0 - 1.0) into exact pixels
                segmentation = []
                x_coords = []
                y_coords = []
                
                for i in range(1, len(parts), 2):
                    x = float(parts[i]) * width
                    y = float(parts[i+1]) * height
                    segmentation.append(x)
                    segmentation.append(y)
                    x_coords.append(x)
                    y_coords.append(y)

                # Calculate Bounding Box (required for COCO format)
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                bbox_w = xmax - xmin
                bbox_h = ymax - ymin

                # 4. Add the polygon to the JSON
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "segmentation": [segmentation],
                    "bbox": [xmin, ymin, bbox_w, bbox_h],
                    "iscrowd": 0,
                    "area": bbox_w * bbox_h
                })
                ann_id += 1
                
        image_id += 1

    # 5. Save the final JSON file
    output_file = "cvat_annotations.json"
    with open(output_file, "w") as f:
        json.dump(coco_data, f)
        
    print(f"Done! Successfully created '{output_file}'.")

if __name__ == "__main__":
    main()