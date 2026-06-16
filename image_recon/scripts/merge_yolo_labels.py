"""
Merge ball and cross YOLO annotations for the SAME images into single label files.

The two datasets annotated the same photos separately:
  - Ball dataset: annotated balls only (class 0=WBall, 1=OBall)
  - Cross dataset: annotated cross only (class 0=x → remapped to 2)

For each image:
  - If both ball and cross labels exist → merge into one file
  - If only ball labels exist → write ball labels with dummy keypoints
  - If only cross labels exist → write cross labels only

Output format per line: class cx cy w h kp1x kp1y kp1v kp2x kp2y kp2v kp3x kp3y kp3v kp4x kp4y kp4v

Merged schema:
  0 = WBall
  1 = OBall
  2 = x (cross)
kpt_shape: [4, 3]
"""

import os
import glob

# ── Configuration ──────────────────────────────────────────────────────────────

PROJECT_ROOT = "."

OUT_LABEL_DIR = os.path.join(PROJECT_ROOT, "image_recon", "YOLO_data", "labels", "train")

BALL_LABEL_DIR  = os.path.join(PROJECT_ROOT, "image_recon", "yolo_data_extra", "balls", "obj_Train_data")
CROSS_LABEL_DIR = os.path.join(PROJECT_ROOT, "image_recon", "yolo_data_extra", "cross", "labels")

# Set True if ball dataset has 0=OBall, 1=WBall (check obj.names)
SWAP_BALL_CLASSES = False

# ── Helpers ────────────────────────────────────────────────────────────────────

DUMMY_KPT = "0 0 0 0 0 0 0 0 0 0 0 0"

def convert_ball_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    if SWAP_BALL_CLASSES:
        cls = 1 - cls
    bbox = " ".join(parts[1:5])
    return f"{cls} {bbox} {DUMMY_KPT}"

def convert_cross_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = 2
    bbox = " ".join(parts[1:5])
    if len(parts) >= 17:
        keypoints = " ".join(parts[5:17])
    else:
        existing = parts[5:]
        pad_count = 12 - len(existing)
        keypoints = " ".join(existing) + (" " + " ".join(["0"] * pad_count) if pad_count > 0 else "")
    return f"{cls} {bbox} {keypoints}"

def read_converted(path, converter):
    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = converter(line)
            if result:
                lines.append(result)
    return lines

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_LABEL_DIR, exist_ok=True)

    # Index all ball and cross label files by stem
    ball_files  = {os.path.splitext(os.path.basename(p))[0]: p
                   for p in glob.glob(os.path.join(BALL_LABEL_DIR,  "**", "*.txt"), recursive=True)}
    cross_files = {os.path.splitext(os.path.basename(p))[0]: p
                   for p in glob.glob(os.path.join(CROSS_LABEL_DIR, "**", "*.txt"), recursive=True)}

    all_stems = set(ball_files) | set(cross_files)
    print(f"Ball label files  : {len(ball_files)}")
    print(f"Cross label files : {len(cross_files)}")
    print(f"Unique image stems: {len(all_stems)}")
    print(f"  Both ball+cross : {len(set(ball_files) & set(cross_files))}")
    print(f"  Ball only       : {len(set(ball_files) - set(cross_files))}")
    print(f"  Cross only      : {len(set(cross_files) - set(ball_files))}")

    written = 0
    skipped = 0

    for stem in sorted(all_stems):
        merged_lines = []

        if stem in ball_files:
            merged_lines += read_converted(ball_files[stem], convert_ball_line)

        if stem in cross_files:
            merged_lines += read_converted(cross_files[stem], convert_cross_line)

        if merged_lines:
            out_path = os.path.join(OUT_LABEL_DIR, stem + ".txt")
            with open(out_path, "w") as f:
                f.write("\n".join(merged_lines) + "\n")
            written += 1
        else:
            skipped += 1

    print(f"\n✅ Done! Written: {written}  Skipped (empty): {skipped}")
    print(f"Output → {OUT_LABEL_DIR}")

    # Sanity check: show a merged file (one that has both balls and cross)
    both = sorted(set(ball_files) & set(cross_files))
    if both:
        sample_path = os.path.join(OUT_LABEL_DIR, both[0] + ".txt")
        print(f"\n── Sample merged label ({both[0]}.txt) ──")
        with open(sample_path) as f:
            print(f.read())

if __name__ == "__main__":
    main()