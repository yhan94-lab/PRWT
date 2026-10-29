import argparse
import logging
import os
import sys
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore")
sys.path.append(".")

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from datasets import brats as brats
from monai.metrics import compute_dice, compute_hausdorff_distance, compute_iou,compute_average_surface_distance

from wetrPRWT.model_attn_aff import WeTr

from texttable import Texttable
parser = argparse.ArgumentParser()
parser.add_argument("--infer_set", default="test", type=str, help="infer_set")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--model_path", default=r"", type=str, help="model_path")
parser.add_argument("--backbone", default="mit_b1", type=str, help="backbone")
parser.add_argument("--data_folder", default=r"", type=str, help="dataset folder")
parser.add_argument( "--list_folder", default=r"", type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=4, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="ignore index")
parser.add_argument("--gpus", type=str, default="0", help="gpu ids, example: 0,1")
parser.add_argument("--log_step", type=int, default=1, help="log interval")
parser.add_argument("--num_workers", type=int, default=4, help="num workers")
parser.add_argument("--stride", default=[4, 2, 2, 1], help="encoder stride")
parser.add_argument("--output_root", default=None, type=str, help="Optional output root for test/preds and test/logs.")

def fix_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

def format_tabs_metric(scores, name_list, cat_list=None):
    _keys_iou = list(scores[0]['iou'].keys())
    _values_iou = []
    _keys_Dice = list(scores[0]['Dice'].keys())
    _values_Dice = []
    _keys_HD95 = list(scores[0]['HD95'].keys())
    _values_HD95 = []
    _keys_ASSD = list(scores[0]['ASSD'].keys())
    _values_ASSD = []


    for i in range(len(scores)):
        _values_iou.append(list(scores[i]['iou'].values()))
        _values_Dice.append(list(scores[i]['Dice'].values()))
        _values_HD95.append(list(scores[i]['HD95'].values()))
        _values_ASSD.append(list(scores[i]['ASSD'].values()))

    _values_iou = np.array(_values_iou) * 100
    _values_Dice = np.array(_values_Dice) * 100
    _values_HD95 = np.array(_values_HD95)
    _values_ASSD = np.array(_values_ASSD)

    t = Texttable()
    t.header(["Class"] + name_list)

    for i in range(len(_keys_iou)):
        if i == 0:
            t.add_row([cat_list[i]] + list(_values_iou[:, i]) + list(_values_Dice[:, i]) + ['None'] * (2 * len(scores)))
        else:
            t.add_row([cat_list[i]] + list(_values_iou[:, i]) + list(_values_Dice[:, i]) + list(_values_HD95[:, i-1]) + list(_values_ASSD[:, i-1]))

    t.add_row(["mean"] + list(_values_iou.mean(1)) + list(_values_Dice.mean(1)) + list(_values_HD95.mean(1))+ list(_values_ASSD.mean(1)) )

    return t.draw()

def to_onehot(labels, num_classes):
    safe_labels = labels.copy()
    safe_labels[(safe_labels < 0) | (safe_labels >= num_classes)] = 0
    return np.eye(num_classes, dtype=np.uint8)[safe_labels].transpose(0, 3, 1, 2)


def foreground_pixel_metrics(preds, gts, num_classes, ignore_index=None):
    valid = np.ones_like(gts, dtype=bool) if ignore_index is None else gts != ignore_index
    pred = preds[valid]
    gt = gts[valid]
    metrics = {}
    total = {"tp": 0, "fp": 0, "fn": 0}
    for cls_idx in range(1, num_classes):
        pred_mask = pred == cls_idx
        gt_mask = gt == cls_idx
        counts = {
            "tp": int(np.logical_and(pred_mask, gt_mask).sum()),
            "fp": int(np.logical_and(pred_mask, ~gt_mask).sum()),
            "fn": int(np.logical_and(~pred_mask, gt_mask).sum()),
        }
        for key in total:
            total[key] += counts[key]
        metrics[brats.class_list[cls_idx]] = counts
    metrics["fg_micro"] = total
    for counts in metrics.values():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        counts["precision"] = tp / float(tp + fp) if tp + fp > 0 else np.nan
        counts["sensitivity"] = tp / float(tp + fn) if tp + fn > 0 else np.nan
    return metrics


def format_segmentation_pixel_metrics(metrics):
    table = Texttable(max_width=120)
    table.set_precision(3)
    table.header(["Class", "Precision (%)", "Sensitivity (%)"])

    for name, values in metrics.items():
        if name == "fg_micro":
            continue

        table.add_row([
            name,
            100.0 * values["precision"],
            100.0 * values["sensitivity"],
        ])

    return table.draw()


def validate(args=None):
    model = WeTr(
        backbone=args.backbone,
        num_classes=args.num_classes,
        embedding_dim=256,
        stride=args.stride,
        pretrained=False,
        pooling=args.pooling,
    )

    trained_state_dict = torch.load(args.model_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(state_dict=new_state_dict, strict=False)
    model.to(device)

    val_dataset = brats.brats2021SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage=args.infer_set,
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_preds = []
    all_gts = []

    model.eval()
    with torch.no_grad():
        for idx, (names, inputs, labels, _) in enumerate(val_loader):
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            img_name = [item + ".png" for item in names]

            _, segs, _, _ = model(inputs)
            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode="bilinear", align_corners=False)

            pred = torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.uint8)
            gt = labels.detach().cpu().numpy().astype(np.uint8)

            all_preds.append(pred)
            all_gts.append(gt)

            pred_for_save = pred.copy()
            pred_for_save[pred_for_save == 3] = 4
            for item, name in zip(pred_for_save, img_name):
                item = Image.fromarray(item, mode="L").convert("P")
                item.putpalette(palette)
                item.save(os.path.join(preds_save_dir, name))

            if (idx + 1) % max(args.log_step, 1) == 0:
                logging.info("idx:{:<5}/{:<5} image_name: {}".format(idx + 1, len(val_dataset), img_name))

        if getattr(args, "save_only", False):
            logging.info("save_only=True: saved predictions; skipped aggregate 2D metrics.")
            return True

        all_preds = np.concatenate(all_preds, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)

        pred_onehot = to_onehot(all_preds, args.num_classes)
        gt_onehot = to_onehot(all_gts, args.num_classes)

        pred_tensor = torch.from_numpy(pred_onehot)
        gt_tensor = torch.from_numpy(gt_onehot)

        dice = compute_dice(pred_tensor, gt_tensor, include_background=True, ignore_empty=False)
        hd95 = compute_hausdorff_distance(
            pred_tensor,
            gt_tensor,
            percentile=95,
            include_background=True,
        )
        assd = compute_average_surface_distance(
            torch.from_numpy(pred_onehot),
            torch.from_numpy(gt_onehot),
            symmetric=True,
            include_background=True,
        )
        assd_valid = torch.isfinite(assd)
        assd_for_mean = torch.where(
            assd_valid,
            assd,
            torch.full_like(assd, float("nan")),
        )
        assd = assd_for_mean

        iou = compute_iou(pred_tensor, gt_tensor, include_background=True, ignore_empty=False)

        results = {
            "Dice": {
                brats.class_list[i]: torch.nanmean(dice, dim=0)[i].item()
                for i in range(0, args.num_classes)
            },
            "HD95": {
                brats.class_list[i]: torch.nanmean(hd95, dim=0)[i].item()
                for i in range(1, args.num_classes)
            },
            "iou": {
                brats.class_list[i]: torch.nanmean(iou, dim=0)[i].item()
                for i in range(0, args.num_classes)
            },
            "ASSD": {
                brats.class_list[i]: torch.nanmean(assd, dim=0)[i].item()
                for i in range(1, args.num_classes)
            },

        }
        tab_results = format_tabs_metric(
            [results],
            ["IOU", "DICE", "HD95","ASSD"],
            cat_list=brats.class_list,
        )
        logging.info("\n" + tab_results)
        pixel_metrics = foreground_pixel_metrics(all_preds, all_gts, args.num_classes, args.ignore_index)
        logging.info("\nSegmentation foreground pixel metrics:\n" + format_segmentation_pixel_metrics(pixel_metrics))

    return True



if __name__ == "__main__":
    args = parser.parse_args()

    base_dir = args.output_root if args.output_root else args.model_path.split("checkpoints")[0]

    preds_save_dir = os.path.join(base_dir, "")
    logs_dir = os.path.join(base_dir, "")
    log_path = os.path.join(base_dir, "")

    colormap_path = os.path.join(args.list_folder, "grayscale_colormap.txt")
    with open(colormap_path, "r", encoding="utf-8") as f:
        palette = eval(f.read().strip())

    for path in (preds_save_dir, logs_dir):
        os.makedirs(path, exist_ok=True)

    if torch.cuda.is_available() and args.gpus != "cpu":
        device = torch.device(f"cuda:{args.gpus}")
    else:
        device = torch.device("cpu")
    fix_seed(1)
    validate(args=args)
