"""
DENTEX → nnU-Net Conversion Script (v2)
========================================
Converts quadrant-enumeration-disease data to nnU-Net v2 format.

Usage:
    python convert_dentex.py \
        --dentex_dir ~/.cache/kagglehub/datasets/truthisneverlinear/dentex-challenge-2023/versions/1/training_data/training_data/quadrant-enumeration-disease \
        --output_dir ~/nnunet/nnUNet_raw/Dataset001_DENTEX

Label Map (from categories_3):
    0 = Impacted
    1 = Caries
    2 = Deep Caries
    3 = Periapical Lesion
    4 = Background (unlabeled teeth)
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
import cv2
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Correct label map from categories_3 ───────────────────────────────────────
LABEL_MAP = {
    0: 1,  # Impacted
    1: 2,  # Caries
    2: 3,  # Deep Caries
    3: 4,  # Periapical Lesion
}

CATEGORY_NAMES = {
    0: 'Impacted',
    1: 'Caries',
    2: 'Deep Caries',
    3: 'Periapical Lesion',
}

DATASET_JSON_TEMPLATE = {
    "channel_names": {"0": "X-Ray"},
    "labels": {
        "background":        "0",
        "Impacted":          "1",
        "Caries":            "2",
        "Deep Caries":       "3",
        "Periapical Lesion": "4",
    },
    "numTraining": 0,
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO"
}


def load_coco_json(json_path: str):
    """Load DENTEX JSON — uses categories_3 for diagnosis labels."""
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"  JSON keys found: {list(data.keys())}")

    # categories_3 = diagnosis labels
    cat_id_to_label = {}
    for cat in data["categories_3"]:
        cid = cat["id"]
        if cid in LABEL_MAP:
            cat_id_to_label[cid] = LABEL_MAP[cid]
            print(f"  Mapped category {cid} ({CATEGORY_NAMES[cid]}) → label {LABEL_MAP[cid]}")
        else:
            print(f"  [WARN] Unknown category id {cid} '{cat['name']}' — skipping")

    # image id → info
    img_id_to_info = {img["id"]: img for img in data["images"]}

    # group annotations by image id
    annotations_by_image = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        annotations_by_image.setdefault(iid, []).append(ann)

    return img_id_to_info, annotations_by_image, cat_id_to_label


def polygon_to_mask(mask: np.ndarray, segmentation: list, label: int) -> np.ndarray:
    """
    Fill a polygon from COCO segmentation into the mask array.
    segmentation is a flat list [x1,y1,x2,y2,...] — convert to (N,2) points.
    """
    for seg in segmentation:
        pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], color=label)
    return mask


def png_to_nifti(image_path: str) -> nib.Nifti1Image:
    """Convert a PNG to grayscale 3D NIfTI (H x W x 1)."""
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    arr = arr[:, :, np.newaxis]
    return nib.Nifti1Image(arr, affine=np.eye(4))


def mask_to_nifti(mask: np.ndarray) -> nib.Nifti1Image:
    """Convert 2D uint8 mask to 3D NIfTI (H x W x 1)."""
    mask_3d = mask[:, :, np.newaxis].astype(np.uint8)
    return nib.Nifti1Image(mask_3d, affine=np.eye(4))


def process_split(ids, img_id_to_info, annotations_by_image, cat_id_to_label,
                  xray_dir, output_dir, img_folder, lbl_folder, split_name):
    print(f"\nProcessing {split_name} ({len(ids)} images) …")
    skipped = 0
    for idx, img_id in enumerate(ids):
        info     = img_id_to_info[img_id]
        filename = info["file_name"]
        h, w     = info["height"], info["width"]

        img_path = xray_dir / filename
        if not img_path.exists():
            print(f"  [WARN] Missing file: {img_path} — skipping")
            skipped += 1
            continue

        # Build polygon mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in annotations_by_image.get(img_id, []):
            label = cat_id_to_label.get(ann["category_id_3"])
            if label is None:
                continue
            if ann.get("segmentation"):
                mask = polygon_to_mask(mask, ann["segmentation"], label)
            else:
                # fallback to bbox if no polygon
                x, y, bw, bh = [int(v) for v in ann["bbox"]]
                mask[y:y+bh, x:x+bw] = label

        case_id = f"DENTEX_{idx:04d}"

        nii_img  = png_to_nifti(str(img_path))
        nii_mask = mask_to_nifti(mask)

        nib.save(nii_img,  output_dir / img_folder / f"{case_id}_0000.nii.gz")
        nib.save(nii_mask, output_dir / lbl_folder / f"{case_id}.nii.gz")

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(ids)} done …")

    print(f"  Done. Skipped: {skipped}")
    return len(ids) - skipped


def convert(dentex_dir: str, output_dir: str, val_split: float = 0.1):
    dentex_dir = Path(dentex_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    xray_dir  = dentex_dir / "xrays"
    json_path = dentex_dir / "train_quadrant_enumeration_disease.json"

    assert xray_dir.exists(),  f"Cannot find xrays folder at {xray_dir}"
    assert json_path.exists(), f"Cannot find JSON at {json_path}"

    print("Loading annotations …")
    img_id_to_info, annotations_by_image, cat_id_to_label = load_coco_json(str(json_path))

    image_ids = list(img_id_to_info.keys())
    train_ids, val_ids = train_test_split(image_ids, test_size=val_split, random_state=42)

    # Create output directories
    for folder in ["imagesTr", "labelsTr", "imagesVal", "labelsVal"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    n_train = process_split(
        train_ids, img_id_to_info, annotations_by_image, cat_id_to_label,
        xray_dir, output_dir, "imagesTr", "labelsTr", "train"
    )
    process_split(
        val_ids, img_id_to_info, annotations_by_image, cat_id_to_label,
        xray_dir, output_dir, "imagesVal", "labelsVal", "val"
    )

    # Write dataset.json
    dataset_json = DATASET_JSON_TEMPLATE.copy()
    dataset_json["numTraining"] = n_train
    dataset_json["name"] = "DENTEX"
    dataset_json["description"] = "Dental pathology semantic segmentation — DENTEX MICCAI 2023"

    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n✅ Done! Dataset written to: {output_dir}")
    print(f"   Training samples:   {n_train}")
    print(f"   Validation samples: {len(val_ids)}")
    print(f"\nNext steps:")
    print(f"  1. nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")
    print(f"  2. nnUNetv2_train 1 2d 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DENTEX to nnU-Net format")
    parser.add_argument("--dentex_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()
    convert(args.dentex_dir, args.output_dir, args.val_split)