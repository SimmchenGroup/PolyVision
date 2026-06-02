import os
import re
from collections import defaultdict

def rename_tiff_files(folder_path, dry_run=True):
    pattern = re.compile(r'^(S\d+-\d+_4X_[\d.]+uL_)(\d+)(\.tiff)$', re.IGNORECASE)

    groups = defaultdict(list)
    for fname in os.listdir(folder_path):
        m = pattern.match(fname)
        if m:
            prefix, index, ext = m.group(1), int(m.group(2)), m.group(3)
            groups[prefix].append((index, fname, ext))

    if not groups:
        print("No matching files found.")
        return

    for prefix, files in groups.items():
        files.sort(key=lambda x: x[0])
        print(f"\nPrefix: {prefix}  ({len(files)} files)")
        renames = []
        for new_idx, (_, fname, ext) in enumerate(files, start=1):
            new_name = f"{prefix}{new_idx:02d}{ext}"
            renames.append((fname, new_name))
            status = "[DRY RUN] " if dry_run else ""
            print(f"  {status}{fname}  ->  {new_name}")

        if not dry_run:
            for old_name, new_name in renames:
                os.rename(
                    os.path.join(folder_path, old_name),
                    os.path.join(folder_path, new_name),
                )
            print(f"  Renamed {len(renames)} files.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rename S##-####_4X_##uL_##.tiff files so the trailing index starts at 01."
    )
    parser.add_argument("folder", help="Path to the folder containing the .tiff files")
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually rename files (default is dry-run preview only)"
    )
    args = parser.parse_args()

    rename_tiff_files(args.folder, dry_run=not args.apply)