"""
Training entry point.
======================
NOTE: The proposed model architecture (config YAML + custom modules) is
withheld pending publication and is not included in this repository yet
(see README.md -> "Proposed architecture release status"). This script
will work as-is with any *stock* Ultralytics model config (e.g. yolo11n.pt,
yolov8n.yaml) for baseline comparisons, and will work with the proposed
architecture as soon as its files are dropped back into this repo
(instructions are included in the withheld bundle).

Usage
-----
    python scripts/train.py --model yolo11n.pt --data configs/fault_seg.yaml
    python scripts/train.py --model yolo11n.pt --data configs/laser.yaml

Once the proposed architecture is released, train it the same way:
    python scripts/train.py --model <path-to-released-model-config>.yaml \\
        --data configs/fault_seg.yaml --name proposed_faultseg
"""

import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train a YOLO model on the railway wheel defect datasets.")
    p.add_argument("--model", type=str, default="yolo11n.pt", help="Model config (.yaml) or checkpoint (.pt) to start from.")
    p.add_argument("--data", type=str, required=True, help="Path to dataset YAML, e.g. configs/fault_seg.yaml")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0", help="'0' for first GPU, 'cpu' for CPU.")
    p.add_argument("--name", type=str, default="train_run")
    p.add_argument("--optimizer", type=str, default="AdamW")
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=float, default=5)
    p.add_argument("--close-mosaic", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        close_mosaic=args.close_mosaic,
    )


if __name__ == "__main__":
    main()
