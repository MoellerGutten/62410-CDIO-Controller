import os
import glob
import math

MAIN_FOLDER   = "image_recon/YOLO_data_6.0"
EXTRA_FOLDER  = "image_recon/yolo_data_extra"
CROSS_CLASS   = 2
DUP_THRESHOLD = 0.02

ZERO_KPT = " " + " ".join(["0.000000"] * 12)


def polygon_to_bbox17(class_id, parts):
    coords   = [float(x) for x in parts]
    x_coords = coords[0::2]
    y_coords = coords[1::2]
    cx = (min(x_coords) + max(x_coords)) / 2
    cy = (min(y_coords) + max(y_coords)) / 2
    w  =  max(x_coords) - min(x_coords)
    h  =  max(y_coords) - min(y_coords)
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}{ZERO_KPT}"


def centre(line_str):
    p = line_str.split()
    return (p[0], float(p[1]), float(p[2]))


def is_duplicate(a, b):
    if a[0] != b[0]:
        return False
    return math.sqrt((a[1]-b[1])**2 + (a[2]-b[2])**2) < DUP_THRESHOLD


def find_extra_txt_files():
    """Walk entire extra folder and collect ALL .txt files (except classes.txt)."""
    found = []
    for root, dirs, files in os.walk(EXTRA_FOLDER):
        for f in files:
            if f.endswith('.txt') and f != 'classes.txt':
                found.append(os.path.join(root, f))
    return found


def debug_structure():
    print(f"\n=== Structure of {EXTRA_FOLDER} ===")
    for root, dirs, files in os.walk(EXTRA_FOLDER):
        level = root.replace(EXTRA_FOLDER, '').count(os.sep)
        print(f"{'  '*level}{os.path.basename(root)}/")
        for f in files:
            print(f"{'  '*(level+1)}{f}")
    print("===\n")


def merge():
    debug_structure()

    # Collect all extra .txt label files, keyed by basename
    extra_txt_files = find_extra_txt_files()
    print(f"Found {len(extra_txt_files)} extra label files:")
    for p in extra_txt_files:
        print(f"  {p}")
    print()

    # Build a map: filename -> path (last one wins if duplicates)
    extra_map = {os.path.basename(p): p for p in extra_txt_files}

    for split in ['train', 'val']:
        main_dir = os.path.join(MAIN_FOLDER, 'labels', split)
        if not os.path.exists(main_dir):
            print(f"Skipping '{split}' — not found.")
            continue

        main_files = sorted(f for f in os.listdir(main_dir) if f.endswith('.txt') and f != 'classes.txt')
        # Also include extra files that don't exist in main (new images)
        all_files = sorted(set(main_files) | set(extra_map.keys()))

        print(f"[{split}]  main={len(main_files)}  extra_available={len(extra_map)}  processing={len(all_files)}")
        added_total = 0

        for fname in all_files:
            final_lines      = []
            accepted_centres = []

            # 1. Load main file as-is (already 17-number pose format)
            main_path = os.path.join(main_dir, fname)
            if os.path.exists(main_path):
                with open(main_path) as f:
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        c = centre(line)
                        if not any(is_duplicate(c, ac) for ac in accepted_centres):
                            accepted_centres.append(c)
                            final_lines.append(line)

            # 2. Load matching extra file and convert
            if fname in extra_map:
                with open(extra_map[fname]) as f:
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        parts = line.split()
                        coord_parts = parts[1:]

                        if len(parts) == 17:
                            parts[0] = str(CROSS_CLASS)
                            new_line = " ".join(parts)
                        elif len(parts) == 5:
                            parts[0] = str(CROSS_CLASS)
                            new_line = " ".join(parts) + ZERO_KPT
                        elif len(coord_parts) >= 4 and len(coord_parts) % 2 == 0:
                            new_line = polygon_to_bbox17(CROSS_CLASS, coord_parts)
                        else:
                            print(f"  WARNING: skipping odd line ({len(parts)} parts) in {fname}")
                            continue

                        c = centre(new_line)
                        if not any(is_duplicate(c, ac) for ac in accepted_centres):
                            accepted_centres.append(c)
                            final_lines.append(new_line)
                            added_total += 1

            # 3. Write back
            if final_lines:
                os.makedirs(main_dir, exist_ok=True)
                with open(main_path, 'w') as f:
                    f.write('\n'.join(final_lines) + '\n')

        print(f"  Extra cross annotations merged: {added_total}\n")

    print("Done.")


if __name__ == "__main__":
    merge()