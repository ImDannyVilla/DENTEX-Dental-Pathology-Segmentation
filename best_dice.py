import nibabel as nib
import numpy as np
import cv2
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True)
parser.add_argument('--labels_dir', type=str, required=True)
args = parser.parse_args()

results_dir = Path(args.results_dir)
labels_dir = Path(args.labels_dir)

def dice_score(pred, gt, label):
    pred_bin = (pred == label)
    gt_bin = (gt == label)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    if total == 0:
        return None
    return 2 * intersection / total

cases = []
for f in sorted(os.listdir(results_dir)):
    if not f.endswith('.nii.gz') or 'DENTEX' not in f:
        continue
    case_id = f.replace('.nii.gz', '')
    gt_path = os.path.join(labels_dir, f)
    pred_path = os.path.join(results_dir, f)
    if not os.path.exists(gt_path):
        continue
    gt   = nib.load(gt_path).get_fdata()[:, :, 0]
    pred = nib.load(pred_path).get_fdata()[:, :, 0]

    # Resize gt to match pred
    gt = cv2.resize(gt.astype(np.float32),
                    (pred.shape[1], pred.shape[0]),
                    interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    dices = [dice_score(pred, gt, l) for l in [1,2,3,4]]
    valid = [d for d in dices if d is not None]
    mean = np.mean(valid) if valid else 0
    cases.append((case_id, mean, dices))

cases.sort(key=lambda x: x[1], reverse=True)
print("Top 5 best predictions:")
for case_id, mean, dices in cases[:5]:
    print(f"  {case_id} | Mean Dice: {mean:.3f} | {[f'{d:.2f}' if d is not None else 'N/A' for d in dices]}")