"""
create_bundle.py
================
Packages the competition programs and pages into bundle.zip,
ready to upload on Codabench.

Usage:
    python tools/create_bundle.py
"""

import os
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(REPO_ROOT, "bundle.zip")

INCLUDE = [
    "competition.yaml",
    "logo.png",
    "ingestion_program/",
    "scoring_program/",
    "pages/",
    "dev_phase/input_data.zip",
    "dev_phase/reference_data.zip",
    "solution/",
    "dev_phase/",
    "scoring_res/",
    "ingestion_res/",
]


def zip_dir(zf: zipfile.ZipFile, dirpath: str, arcbase: str):
    for root, dirs, files in os.walk(dirpath):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for fname in files:
            if fname.endswith(".pyc"):
                continue
            full = os.path.join(root, fname)
            arc = os.path.join(arcbase, os.path.relpath(full, dirpath))
            zf.write(full, arc)
            print(f"  + {arc}")


def main():
    with zipfile.ZipFile(BUNDLE_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE:
            full = os.path.join(REPO_ROOT, item.rstrip("/"))
            if os.path.isdir(full):
                zip_dir(zf, full, item.rstrip("/"))
            elif os.path.isfile(full):
                zf.write(full, item)
                print(f"  + {item}")
            else:
                print(f"  [WARNING] Not found, skipping: {full}")

    size = os.path.getsize(BUNDLE_PATH) / 1024
    print(f"\nBundle created: {BUNDLE_PATH}  ({size:.1f} KB)")
    print("Upload bundle.zip on Codabench -> Create Competition -> Upload bundle.")


if __name__ == "__main__":
    main()
