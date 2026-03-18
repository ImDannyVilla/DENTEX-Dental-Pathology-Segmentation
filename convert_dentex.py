"""
DENTEX → nnU-Net Conversion Script
====================================
Converts quadrant-enumeration-disease data to nnU-Net v2 format.

Usage:
    python convert_dentex.py --dentex_dir "C:/path/to/quadrant-enumeration-disease"
                             --output_dir "C:/nnunet/nnUNet_raw/Dataset001_DENTEX"

Label Map:
    0 = Background
    1 = Caries
    2 = Deep Caries
    3 = Periapical Lesion
    4 = Impacted Tooth
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split


# ── Category name → label index ───────────────────────────────────────────────
# Adjust these if the DENTEX JSON uses different category names
LABEL_MAP = {
    "background":         0,
    "Caries":             1,
    "Deep Caries":        2,
    "Periapical Lesion":  3,
    "Impacted Tooth":     4,
}

DATASET_JSON_TEMPLATE = {
    "channel_names": {"0": "X-Ray"},
    "labels": {
        "background":        "0",
        "Caries":            "1",
        "Deep Caries":       "2",
        "Periapical Lesion": "3",
        "Impacted Tooth":    "4",
    },
    "numTraining": 0,           # filled in at runtime
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO"
}


def load_coco_json(json_path: str):
    """Load COCO-style annotation JSON and build lookup dicts."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Build category id → label index
    cat_id_to_label = {}
    for cat in data["categories"]:
        name = cat["name"]
        if name in LABEL_MAP:
            cat_id_to_label[cat["id"]] = LABEL_MAP[name]
        else:
            print(f"  [WARN] Unknown category '{name}' — skipping")

    # Build image id → filename
    img_id_to_info = {img["id"]: img for img in data["images"]}

    # Group annotations by image id
    annotations_by_image = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        annotations_by_image.setdefault(iid, []).append(ann)

    return img_id_to_info, annotations_by_image, cat_id_to_label


def bbox_to_mask(mask: np.ndarray, bbox, label: int):
    """
    Paint a bounding-box region with the given label onto the mask array.
    COCO bbox format: [x, y, width, height]
    """
    x, y, w, h = [int(v) for v in bbox]
    x2 = min(x + w, mask.shape[1])
    y2 = min(y + h, mask.shape[0])
    mask[y:y2, x:x2] = label
    return mask


def png_to_nifti(image_path: str) -> nib.Nifti1Image:
    """Convert a grayscale/RGB PNG to a 3-D NIfTI (H x W x 1)."""
    img = Image.open(image_path).convert("L")   # force grayscale
    arr = np.array(img, dtype=np.float32)
    arr = arr[:, :, np.newaxis]                  # add channel dim
    nii = nib.Nifti1Image(arr, affine=np.eye(4))
    return nii


def mask_to_nifti(mask: np.ndarray) -> nib.Nifti1Image:
    """Convert a 2-D uint8 mask to a 3-D NIfTI (H x W x 1)."""
    mask_3d = mask[:, :, np.newaxis].astype(np.uint8)
    nii = nib.Nifti1Image(mask_3d, affine=np.eye(4))
    return nii


def convert(dentex_dir: str, output_dir: str, val_split: float = 0.1):
    dentex_dir = Path(dentex_dir)
    output_dir = Path(output_dir)

    xray_dir  = dentex_dir / "xrays"
    json_path = dentex_dir / "train_quadrant_enumeration_disease.json"

    assert xray_dir.exists(),  f"Cannot find xrays folder at {xray_dir}"
    assert json_path.exists(), f"Cannot find JSON at {json_path}"

    print("Loading annotations …")
    img_id_to_info, annotations_by_image, cat_id_to_label = load_coco_json(str(json_path))

    image_ids = list(img_id_to_info.keys())
    train_ids, val_ids = train_test_split(image_ids, test_size=val_split, random_state=42)

    # Create output directories
    for split in ["imagesTr", "labelsTr", "imagesVal", "labelsVal"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    def process_split(ids, img_folder, lbl_folder, split_name):
        print(f"\nProcessing {split_name} ({len(ids)} images) …")
        for idx, img_id in enumerate(ids):
            info     = img_id_to_info[img_id]
            filename = info["file_name"]
            h, w     = info["height"], info["width"]

            img_path = xray_dir / filename
            if not img_path.exists():
                print(f"  [WARN] Missing file: {img_path} — skipping")
                continue

            # Build label mask
            mask = np.zeros((h, w), dtype=np.uint8)
            for ann in annotations_by_image.get(img_id, []):
                label = cat_id_to_label.get(ann["category_id"])
                if label is None:
                    continue
                mask = bbox_to_mask(mask, ann["bbox"], label)

            # nnU-Net naming convention: CASE_XXXX_0000.nii.gz (image), CASE_XXXX.nii.gz (label)
            case_id = f"DENTEX_{idx:04d}"

            nii_img  = png_to_nifti(str(img_path))
            nii_mask = mask_to_nifti(mask)

            nib.save(nii_img,  output_dir / img_folder / f"{case_id}_0000.nii.gz")
            nib.save(nii_mask, output_dir / lbl_folder / f"{case_id}.nii.gz")

            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{len(ids)} done …")

        return len(ids)

    n_train = process_split(train_ids, "imagesTr", "labelsTr",   "train")
    process_split(val_ids,   "imagesVal", "labelsVal", "val")

    # Write dataset.json
    dataset_json = DATASET_JSON_TEMPLATE.copy()
    dataset_json["numTraining"] = n_train
    dataset_json["name"] = "DENTEX"
    dataset_json["description"] = "Dental pathology semantic segmentation from DENTEX MICCAI 2023"

    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n✅ Done! nnU-Net dataset written to: {output_dir}")
    print(f"   Training samples : {n_train}")
    print(f"   Validation samples: {len(val_ids)}")
    print(f"\nNext steps:")
    print(f"   1. nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")
    print(f"   2. nnUNetv2_train 1 2d 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DENTEX dataset to nnU-Net format")
    parser.add_argument(
        "--dentex_dir",
        required=True,
        help="Path to quadrant-enumeration-disease folder"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to nnUNet_raw/Dataset001_DENTEX"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation (default: 0.1)"
    )
    args = parser.parse_args()
    convert(args.dentex_dir, args.output_dir, args.val_split)