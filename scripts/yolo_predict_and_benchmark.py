"""
YOLO Detection — Prediction + Full Benchmark Suite
====================================================
Single-model pipeline for a CUSTOM YOLO detection model (best.pt).

What this script does
----------------------
1. PREDICTION
   Runs the model on every image inside `raw/` and saves annotated
   images (bounding boxes drawn) into `outputs/predictions/`.

2. ACCURACY METRICS  (Precision, Recall, mAP50, mAP50-95)
   Runs `model.val()` on `test/images` + `test/labels` (standard YOLO
   format) to get accuracy metrics. A dataset YAML is auto-generated
   for you — you only need to supply the class names once.

3. SPEED METRICS  (Inference ms, Throughput img/s)
   Times REAL inference on the images inside `test/images` (not dummy
   tensors) — once on GPU (if available) and once on CPU, so both
   columns of your table are filled from one run.

4. MODEL SIZE METRICS  (FLOPs, Params)
   Pulled from Ultralytics' built-in `model.info()`.

5. EFFICIENCY RATIO (ER)
   ER = mAP@0.5 * Throughput(img/s) / FLOPs(M)
   computed separately for CPU and GPU throughput.

6. FINAL TABLE
   Everything is assembled into exactly the table you asked for:

   Model | Precision | Recall | mAP50 | mAP50-95 | Inference(ms) GPU |
   Inference(ms) CPU | Throughput(img/s) GPU | Throughput(img/s) CPU |
   FLOPs(M) | Params(M) | ER CPU | ER GPU

   Saved as both a .csv and a .json, plus printed nicely to console.

Expected folder layout (all paths are configurable via CLI flags,
these are just the defaults)
----------------------------------------------------------------
project/
├── best.pt                  <- your trained model
├── raw/                     <- folder of images to run prediction on
│   ├── img1.jpg
│   └── ...
├── test/                    <- folder used to compute accuracy metrics
│   ├── images/
│   │   ├── img1.jpg
│   │   └── ...
│   └── labels/
│       ├── img1.txt         <- YOLO-format label files (same basename)
│       └── ...
└── yolo_predict_and_benchmark.py   <- this script

Usage
-----
    python yolo_predict_and_benchmark.py \\
        --weights best.pt \\
        --raw-dir raw \\
        --test-dir test \\
        --classes fracture spot

If you already have a dataset YAML (e.g. laser.yaml), pass it directly
and skip --classes:

    python yolo_predict_and_benchmark.py --weights best.pt \\
        --raw-dir raw --test-dir test --data-yaml laser.yaml

Run `python yolo_predict_and_benchmark.py --help` for every option.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Run prediction + full benchmark/metrics suite for a YOLO detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", type=str, default="best.pt", help="Path to your trained model (best.pt)")
    p.add_argument("--raw-dir", type=str, default="raw", help="Folder of images to run prediction on")
    p.add_argument("--test-dir", type=str, default="test",
                    help="Folder containing test/images and test/labels (YOLO format) for accuracy metrics")
    p.add_argument("--data-yaml", type=str, default=None,
                    help="Path to an existing dataset YAML. If omitted, one is auto-generated from --test-dir + --classes.")
    p.add_argument("--classes", type=str, nargs="*", default=None,
                    help="Class names in index order, e.g. --classes fracture spot. "
                         "Required if --data-yaml is not given and the YAML can't be inferred.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for prediction on raw/")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS during prediction on raw/")
    p.add_argument("--batch", type=int, default=1, help="Batch size used for model.val()")
    p.add_argument("--warmup", type=int, default=10, help="Warm-up inference runs before timing starts")
    p.add_argument("--timing-runs", type=int, default=None,
                    help="Number of images to time for latency/throughput. "
                         "Default = all images in test/images (capped at 200 for very large sets).")
    p.add_argument("--output-dir", type=str, default="outputs", help="Where to save predictions + metrics results")
    p.add_argument("--skip-cpu", action="store_true", help="Skip CPU timing (only benchmark GPU)")
    p.add_argument("--skip-gpu", action="store_true", help="Skip GPU timing (only benchmark CPU)")
    p.add_argument("--skip-predict", action="store_true", help="Skip running prediction on raw/ (metrics only)")
    p.add_argument("--device-val", type=str, default=None,
                    help="Device to use for model.val() accuracy metrics, e.g. '0' or 'cpu'. "
                         "Default: GPU if available, else CPU.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def ensure_dataset_yaml(args, project_root: Path) -> Path:
    """Return a path to a usable dataset YAML, auto-generating one if needed."""
    if args.data_yaml:
        yaml_path = Path(args.data_yaml)
        if not yaml_path.exists():
            sys.exit(f"❌ --data-yaml given but file not found: {yaml_path}")
        print(f"✅ Using existing dataset YAML: {yaml_path}")
        return yaml_path

    test_dir = Path(args.test_dir).resolve()
    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        sys.exit(
            f"❌ Expected '{images_dir}' and '{labels_dir}' to both exist.\n"
            f"   Your --test-dir must contain an 'images/' and 'labels/' subfolder."
        )

    # Try to infer class names from existing label files + class names file,
    # otherwise require --classes.
    names = args.classes
    names_txt = test_dir / "classes.txt"
    if names is None and names_txt.exists():
        names = [line.strip() for line in names_txt.read_text().splitlines() if line.strip()]
        print(f"✅ Loaded class names from {names_txt}: {names}")

    if names is None:
        # Infer number of classes from label files, but names are unknown.
        max_idx = -1
        for lbl in labels_dir.glob("*.txt"):
            for line in lbl.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    cls_idx = int(float(line.split()[0]))
                    max_idx = max(max_idx, cls_idx)
                except (ValueError, IndexError):
                    continue
        if max_idx < 0:
            sys.exit(
                "❌ No class names provided (--classes) and none could be inferred from labels.\n"
                "   Please pass e.g. --classes fracture spot"
            )
        names = [f"class_{i}" for i in range(max_idx + 1)]
        print(
            f"⚠️  No --classes given. Inferred {len(names)} class(es) from label files "
            f"but using placeholder names: {names}\n"
            f"   (Pass --classes <name1> <name2> ... for correct class names.)"
        )
    else:
        print(f"✅ Using class names: {names}")

    yaml_path = project_root / "auto_generated_dataset.yaml"
    yaml_content = (
        f"# Auto-generated by yolo_predict_and_benchmark.py\n"
        f"path: {test_dir.parent.as_posix()}\n"
        f"train: {test_dir.name}/images\n"
        f"val: {test_dir.name}/images\n"
        f"test: {test_dir.name}/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    yaml_path.write_text(yaml_content)
    print(f"✅ Auto-generated dataset YAML -> {yaml_path}")
    print(f"   (train/val both point at test/images since we only need val()/test metrics here)")
    return yaml_path


def safe_round(x, n=4):
    return round(float(x), n) if x is not None else None


# ──────────────────────────────────────────────────────────────────────────
# Core measurement functions
# ──────────────────────────────────────────────────────────────────────────


def compute_flops_and_params(model, imgsz):
    """Use Ultralytics' built-in profiler. Returns (FLOPs_in_millions, params_in_millions)."""
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    try:
        gflops = get_flops(model.model, imgsz=imgsz)  # returns GFLOPs (billions); 0.0 if thop missing/failed
    except Exception as e:
        print(f"  [WARN] get_flops failed ({e}), trying model.info() fallback")
        gflops = None

    n_params = get_num_params(model.model)

    if not gflops:  # None or 0.0 (thop not installed, or profiling failed)
        try:
            info = model.info(verbose=False, imgsz=imgsz)
            if isinstance(info, (list, tuple)) and len(info) >= 4 and info[3]:
                gflops = info[3]
        except Exception as e:
            print(f"  [WARN] model.info() fallback also failed: {e}")
        if not gflops:
            print("  [WARN] Could not compute FLOPs (is 'thop'/'ultralytics-thop' installed?). Reporting as 0.")
            gflops = 0.0

    flops_m = gflops * 1000.0  # GFLOPs -> "FLOPs (M)" as commonly tabulated in YOLO papers
    params_m = n_params / 1e6
    return flops_m, params_m


def measure_real_inference(weights_path, image_paths, device, imgsz, warmup, n_runs):
    """
    Time REAL inference (full predict pipeline: preprocess + forward + NMS)
    on actual images from disk, on a given device ('cpu' or 'cuda:0' style).

    Returns dict: mean_ms, std_ms, min_ms, max_ms, throughput_img_s
    """
    import torch

    if not image_paths:
        return None

    # Fresh model instance per device to avoid any cross-device state issues.
    model = load_yolo_model(weights_path)
    model.to(device)

    n_runs = min(n_runs, len(image_paths)) if n_runs else len(image_paths)
    timing_imgs = image_paths[:n_runs] if n_runs <= len(image_paths) else (
        image_paths * (n_runs // len(image_paths) + 1)
    )[:n_runs]

    is_cuda = device.startswith("cuda")

    # Warm-up (not timed) — critical for fair GPU numbers (CUDA context / cuDNN autotune)
    warm_imgs = (image_paths * (warmup // len(image_paths) + 1))[:warmup]
    for img in warm_imgs:
        with torch.no_grad():
            model.predict(source=str(img), imgsz=imgsz, device=device, verbose=False, save=False)
    if is_cuda:
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for img in timing_imgs:
            if is_cuda:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.predict(source=str(img), imgsz=imgsz, device=device, verbose=False, save=False)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            else:
                t0 = time.perf_counter()
                model.predict(source=str(img), imgsz=imgsz, device=device, verbose=False, save=False)
                t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times = np.array(times)
    mean_ms = float(times.mean())
    return {
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(float(times.std()), 3),
        "min_ms": round(float(times.min()), 3),
        "max_ms": round(float(times.max()), 3),
        "n_images_timed": len(timing_imgs),
        "throughput_img_s": round(1000.0 / mean_ms, 3),
    }


def load_yolo_model(weights_path):
    """
    Load a YOLO model, with a friendly error if the checkpoint was trained
    with custom architecture modules that aren't registered in the
    currently-installed `ultralytics` package.
    """
    from ultralytics import YOLO

    try:
        return YOLO(str(weights_path))
    except AttributeError as e:
        msg = str(e)
        if "Can't get attribute" in msg or "module 'ultralytics" in msg:
            sys.exit(
                f"\n❌ Failed to load '{weights_path}':\n   {msg}\n\n"
                f"This means the checkpoint was trained with a CUSTOM/modified Ultralytics\n"
                f"(custom blocks/attention modules that are not part of the stock\n"
                f"'pip install ultralytics' package).\n\n"
                f"Fix: install YOUR custom ultralytics fork/repo into this environment\n"
                f"instead of (or before) the stock package, e.g. from the repo root:\n\n"
                f"    pip uninstall ultralytics -y\n"
                f"    pip install -e /path/to/your/custom/ultralytics/repo\n\n"
                f"Then re-run this script in the SAME Python environment.\n"
            )
        raise


def run_prediction_on_raw(weights_path, raw_dir, out_dir, device, imgsz, conf, iou):
    """Run prediction on every image in raw_dir, save annotated images to out_dir."""
    images = find_images(Path(raw_dir))
    if not images:
        print(f"⚠️  No images found in '{raw_dir}', skipping prediction step.")
        return 0

    model = load_yolo_model(weights_path)
    model.predict(
        source=str(raw_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=True,
        project=str(Path(out_dir).parent),
        name=Path(out_dir).name,
        exist_ok=True,
        verbose=False,
    )
    print(f"✅ Saved {len(images)} annotated prediction(s) -> {out_dir}")
    return len(images)


def run_validation(weights_path, data_yaml, device, imgsz, batch):
    """Run model.val() -> Precision, Recall, mAP50, mAP50-95 (box metrics, detection)."""
    model = load_yolo_model(weights_path)
    results = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        split="test",
        device=device,
        verbose=False,
        plots=False,
        save=False,
        save_json=False,
    )
    metrics = results.results_dict

    def get(d, *keys):
        for k in keys:
            if k in d:
                return safe_round(d[k])
        return None

    return {
        "precision": get(metrics, "metrics/precision(B)"),
        "recall": get(metrics, "metrics/recall(B)"),
        "mAP50": get(metrics, "metrics/mAP50(B)"),
        "mAP50_95": get(metrics, "metrics/mAP50-95(B)"),
        "raw": {k: safe_round(v) for k, v in metrics.items() if isinstance(v, (int, float))},
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    try:
        import torch
    except ImportError:
        sys.exit(
            "❌ PyTorch is not installed. Install requirements first:\n"
            "   pip install -r requirements.txt"
        )

    weights_path = Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"❌ Model weights not found: {weights_path}")

    project_root = Path(".").resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_dir = out_dir / "predictions"

    gpu_available = torch.cuda.is_available()
    model_name = weights_path.stem

    print("=" * 70)
    print("  YOLO DETECTION — PREDICTION + BENCHMARK SUITE")
    print("=" * 70)
    print(f"  Model        : {weights_path}")
    print(f"  PyTorch      : {torch.__version__}")
    print(f"  GPU available: {gpu_available}" + (f"  ({torch.cuda.get_device_name(0)})" if gpu_available else ""))
    print(f"  Platform     : {platform.platform()}")
    print("=" * 70)

    # ── 1. Prediction on raw/ ────────────────────────────────────────────
    if not args.skip_predict:
        print("\n[STEP 1/4] Running prediction on raw/ images ...")
        pred_device = "0" if gpu_available else "cpu"
        run_prediction_on_raw(
            weights_path, args.raw_dir, pred_out_dir, pred_device, args.imgsz, args.conf, args.iou
        )
    else:
        print("\n[STEP 1/4] Skipped (--skip-predict).")

    # ── 2. Dataset YAML for accuracy metrics ─────────────────────────────
    print("\n[STEP 2/4] Preparing dataset YAML for accuracy metrics ...")
    data_yaml = ensure_dataset_yaml(args, project_root)

    # ── 3. Accuracy metrics (Precision/Recall/mAP) ───────────────────────
    print("\n[STEP 3/4] Running validation for accuracy metrics (Precision, Recall, mAP50, mAP50-95) ...")
    val_device = args.device_val or ("0" if gpu_available else "cpu")
    val_metrics = run_validation(weights_path, data_yaml, val_device, args.imgsz, args.batch)
    print(f"  Precision : {val_metrics['precision']}")
    print(f"  Recall    : {val_metrics['recall']}")
    print(f"  mAP50     : {val_metrics['mAP50']}")
    print(f"  mAP50-95  : {val_metrics['mAP50_95']}")

    # ── 4. FLOPs / Params ─────────────────────────────────────────────────
    print("\n[STEP 4/4] Computing FLOPs, Params, and timing real inference (CPU + GPU) ...")
    info_model = load_yolo_model(weights_path)
    flops_m, params_m = compute_flops_and_params(info_model, args.imgsz)
    print(f"  FLOPs (M) : {flops_m:.2f}")
    print(f"  Params (M): {params_m:.4f}")

    # ── 5. Real-image latency / throughput on CPU and GPU ────────────────
    test_images = find_images(Path(args.test_dir) / "images")
    if not test_images:
        sys.exit(f"❌ No images found in {Path(args.test_dir) / 'images'} for timing.")

    n_runs = args.timing_runs or min(len(test_images), 200)

    gpu_timing = None
    if gpu_available and not args.skip_gpu:
        print(f"\n  Timing on GPU (cuda:0), {n_runs} real image(s), {args.warmup} warm-up runs ...")
        gpu_timing = measure_real_inference(weights_path, test_images, "cuda:0", args.imgsz, args.warmup, n_runs)
        print(f"    Mean latency : {gpu_timing['mean_ms']} ms   Throughput: {gpu_timing['throughput_img_s']} img/s")
    elif not gpu_available:
        print("\n  ⚠️  No GPU detected — GPU columns will be reported as N/A.")
    else:
        print("\n  Skipped GPU timing (--skip-gpu).")

    cpu_timing = None
    if not args.skip_cpu:
        print(f"\n  Timing on CPU, {n_runs} real image(s), {args.warmup} warm-up runs ...")
        cpu_timing = measure_real_inference(weights_path, test_images, "cpu", args.imgsz, args.warmup, n_runs)
        print(f"    Mean latency : {cpu_timing['mean_ms']} ms   Throughput: {cpu_timing['throughput_img_s']} img/s")
    else:
        print("\n  Skipped CPU timing (--skip-cpu).")

    # ── 6. Efficiency Ratio ───────────────────────────────────────────────
    # ER = mAP@0.5 * Throughput(img/s) / FLOPs(M)
    map50 = val_metrics["mAP50"] or 0.0

    def compute_er(throughput):
        if throughput is None or flops_m in (None, 0):
            return None
        return round((map50 * throughput) / flops_m, 6)

    er_cpu = compute_er(cpu_timing["throughput_img_s"] if cpu_timing else None)
    er_gpu = compute_er(gpu_timing["throughput_img_s"] if gpu_timing else None)

    # ── 7. Assemble final table row ───────────────────────────────────────
    final_row = {
        "Model": model_name,
        "Precision": val_metrics["precision"],
        "Recall": val_metrics["recall"],
        "mAP50": val_metrics["mAP50"],
        "mAP50-95": val_metrics["mAP50_95"],
        "Inference (ms) GPU": gpu_timing["mean_ms"] if gpu_timing else None,
        "Inference (ms) CPU": cpu_timing["mean_ms"] if cpu_timing else None,
        "Throughput (img/s) GPU": gpu_timing["throughput_img_s"] if gpu_timing else None,
        "Throughput (img/s) CPU": cpu_timing["throughput_img_s"] if cpu_timing else None,
        "FLOPs (M)": round(flops_m, 3),
        "Params (M)": round(params_m, 4),
        "ER CPU": er_cpu,
        "ER GPU": er_gpu,
    }

    full_results = {
        "model": model_name,
        "weights_path": str(weights_path),
        "config": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "batch": args.batch,
            "warmup": args.warmup,
            "timing_runs": n_runs,
        },
        "environment": {
            "torch_version": torch.__version__,
            "gpu_available": gpu_available,
            "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
            "platform": platform.platform(),
        },
        "accuracy": val_metrics,
        "architecture": {"FLOPs_M": round(flops_m, 3), "Params_M": round(params_m, 4)},
        "latency_gpu": gpu_timing,
        "latency_cpu": cpu_timing,
        "efficiency_ratio": {"cpu": er_cpu, "gpu": er_gpu},
        "final_table_row": final_row,
    }

    # ── 8. Save outputs ───────────────────────────────────────────────────
    json_path = out_dir / "metrics_results.json"
    json_path.write_text(json.dumps(full_results, indent=2, default=str))

    csv_path = out_dir / "metrics_summary.csv"
    headers = list(final_row.keys())
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        f.write(",".join("" if final_row[h] is None else str(final_row[h]) for h in headers) + "\n")

    # ── 9. Print final table ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL METRICS TABLE")
    print("=" * 70)
    for k, v in final_row.items():
        v_str = "N/A" if v is None else v
        print(f"  {k:<26}: {v_str}")
    print("=" * 70)
    print(f"\n✅ Saved JSON   -> {json_path.resolve()}")
    print(f"✅ Saved CSV    -> {csv_path.resolve()}")
    if not args.skip_predict:
        print(f"✅ Saved images -> {pred_out_dir.resolve()}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
