"""
Rename all images in a folder to pic_NNN.<ext>, continuing a sequence
from a given start number.

Use case:
  Image_model_10 already contains pic_001 ... pic_607
  "More data for model_11" has images with arbitrary names
  -> this script renames them to pic_608, pic_609, ... in place

Usage:
  Just put this script directly inside the folder of images you want to
  rename (e.g. inside "More data for model_11"), then run it with no
  arguments:

      python rename_to_pic_sequence.py

  It will operate on the folder it's sitting in.

  You can still point it at a different folder if you want:

      python rename_to_pic_sequence.py "/path/to/some/other/folder"

By default it does a DRY RUN (just prints what it would do).
Add --apply to actually rename the files.

Sorting:
  Files are sorted by modification time by default (closest to capture order).
  Use --sort name to sort alphabetically instead.
"""

import os
import sys
import argparse

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(folder):
    files = []
    for entry in os.scandir(folder):
        if entry.is_file():
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in IMAGE_EXTS:
                files.append(entry.path)
    return files


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Rename images to pic_NNN sequence.")
    parser.add_argument("folder", nargs="?", default=script_dir,
                         help="Folder containing the images to rename "
                              "(default: the folder this script is in)")
    parser.add_argument("--start", type=int, default=608,
                         help="First number in the sequence (default: 608)")
    parser.add_argument("--width", type=int, default=3,
                         help="Zero-padding width for the number (default: 3, e.g. pic_008)")
    parser.add_argument("--sort", choices=["mtime", "name"], default="mtime",
                         help="Sort order before assigning numbers (default: mtime)")
    parser.add_argument("--apply", action="store_true",
                         help="Actually perform the rename (otherwise dry run only)")
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a directory")
        sys.exit(1)

    files = find_images(folder)
    if not files:
        print("No image files found in that folder.")
        sys.exit(0)

    if args.sort == "mtime":
        files.sort(key=lambda p: os.path.getmtime(p))
    else:
        files.sort(key=lambda p: os.path.basename(p).lower())

    print(f"Found {len(files)} images in: {folder}")
    print(f"Sequence will run from pic_{args.start:0{args.width}d} "
          f"to pic_{args.start + len(files) - 1:0{args.width}d}\n")

    # Build the rename plan first
    plan = []
    for i, src in enumerate(files):
        ext = os.path.splitext(src)[1].lower()
        new_name = f"pic_{args.start + i:0{args.width}d}{ext}"
        dst = os.path.join(folder, new_name)
        plan.append((src, dst))

    # Check for collisions with existing files not part of the rename set
    src_set = set(files)
    for _, dst in plan:
        if os.path.exists(dst) and dst not in src_set:
            print(f"Collision detected: '{dst}' already exists and isn't one of the "
                  f"files being renamed. Aborting to avoid overwriting data.")
            sys.exit(1)

    for src, dst in plan:
        print(f"  {os.path.basename(src)}  ->  {os.path.basename(dst)}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to actually rename the files.")
        return

    # Two-phase rename to avoid clobbering files when names overlap
    temp_names = []
    for src, dst in plan:
        tmp = src + ".tmp_rename"
        os.rename(src, tmp)
        temp_names.append(tmp)

    for tmp, (_, dst) in zip(temp_names, plan):
        os.rename(tmp, dst)

    print(f"\nDone. Renamed {len(plan)} files.")


if __name__ == "__main__":
    main()