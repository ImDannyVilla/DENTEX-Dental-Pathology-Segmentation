"""
Usage:
    python visualize.py \
        --case DENTEX_0022 \
        --imagesTr ~/nnunet/nnUNet_raw/Dataset001_DENTEX/imagesTr \
        --labelsTr ~/nnunet/nnUNet_raw/Dataset001_DENTEX/labelsTr \
        --predictions ~/Projects/dentex-seg/dentex_output \
        --output ~/Projects/dentex-seg/assets/comparison.png

Label Map:
    1 = Impacted       (Blue)
    2 = Caries         (Green)
    3 = Deep Caries    (Red)
    4 = Periapical     (Yellow)
"""

import argparse
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path


COLORS = {
    1: (255, 0, 0),    # Impacted — blue
    2: (0, 255, 0),    # Caries — green
    3: (0, 0, 255),    # Deep Caries — red
    4: (0, 255, 255),  # Periapical — yellow
}


def apply_overlay(xray_bgr, mask):
    mask_resized = cv2.resize(
        mask.astype(np.float32),
        (xray_bgr.shape[1], xray_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)
    overlay = xray_bgr.copy()
    for label, color in COLORS.items():
        overlay[mask_resized == label] = color
    return cv2.addWeighted(xray_bgr, 0.7, overlay, 0.3, 0)


def add_label(img, text):
    return cv2.putText(
        img.copy(), text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2
    )


def resize_h(img, h):
    ratio = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * ratio), h))


def main():
    parser = argparse.ArgumentParser(description="Visualize DENTEX segmentation results")
    parser.add_argument('--case',        type=str, required=True,  help="Case ID e.g. DENTEX_0022")
    parser.add_argument('--imagesTr',    type=str, required=True,  help="Path to imagesTr folder")
    parser.add_argument('--labelsTr',    type=str, required=True,  help="Path to labelsTr folder")
    parser.add_argument('--predictions', type=str, required=True,  help="Path to prediction output folder")
    parser.add_argument('--output',      type=str, required=True,  help="Output PNG path")
    parser.add_argument('--height',      type=int, default=800,    help="Output image height in pixels")
    args = parser.parse_args()

    imagesTr    = Path(args.imagesTr).expanduser()
    labelsTr    = Path(args.labelsTr).expanduser()
    predictions = Path(args.predictions).expanduser()
    output      = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    xray_path = imagesTr    / f"{args.case}_0000.nii.gz"
    gt_path   = labelsTr    / f"{args.case}.nii.gz"
    pred_path = predictions / f"{args.case}.nii.gz"

    for p in [xray_path, gt_path, pred_path]:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    # Load
    xray = nib.load(xray_path).get_fdata()[:, :, 0]
    xray = ((xray - xray.min()) / (xray.max() - xray.min()) * 255).astype(np.uint8)
    xray_bgr = cv2.cvtColor(xray, cv2.COLOR_GRAY2BGR)

    gt_mask   = nib.load(gt_path).get_fdata()[:, :, 0]
    pred_mask = nib.load(pred_path).get_fdata()[:, :, 0]

    print(f"Xray shape:     {xray_bgr.shape}")
    print(f"GT mask shape:  {gt_mask.shape}")
    print(f"Pred mask shape:{pred_mask.shape}")

    # Build overlays
    gt_overlay   = apply_overlay(xray_bgr.copy(), gt_mask)
    pred_overlay = apply_overlay(xray_bgr.copy(), pred_mask)

    # Rotate
    xray_rot = cv2.rotate(xray_bgr,    cv2.ROTATE_90_CLOCKWISE)
    gt_rot   = cv2.rotate(gt_overlay,  cv2.ROTATE_90_CLOCKWISE)
    pred_rot = cv2.rotate(pred_overlay, cv2.ROTATE_90_CLOCKWISE)

    # Labels
    xray_rot = add_label(xray_rot, 'X-Ray')
    gt_rot   = add_label(gt_rot,   'Ground Truth')
    pred_rot = add_label(pred_rot, 'Prediction')

    # Resize and combine
    h = args.height
    xray_rot = resize_h(xray_rot, h)
    gt_rot   = resize_h(gt_rot, h)
    pred_rot = resize_h(pred_rot, h)

    combined = np.hstack([xray_rot, gt_rot, pred_rot])
    cv2.imwrite(str(output), combined)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()