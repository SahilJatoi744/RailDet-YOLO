"""
Comprehensive Metrics Extractor for YOLO Segmentation Models
Extracts: GFLOPs, Parameters, Latency, Precision, Recall, mAP50, mAP50-95, Accuracy
Models: fault-seg & laser (best.pt files)
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_DIR = Path("model_parameters")
MODELS = {
    "fault-seg": MODEL_DIR / "fault-seg.pt",  # 4 classes: Cracks-Scratches, Discloration, Shelling, Wheel
    "laser":     MODEL_DIR / "laser.pt",       # 2 classes: fracture, spot
}
IMG_SIZE   = 640   # standard inference size
WARMUP     = 10    # warm-up runs before latency timing
TIMING_RUNS = 100  # runs used for latency measurement
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

print(f"\n{'='*65}")
print(f"  YOLO Model Metrics Extractor")
print(f"  Device : {DEVICE.upper()}")
print(f"  PyTorch: {torch.__version__}")
print(f"{'='*65}\n")


def check_model_files():
    """Verify both .pt files exist before proceeding."""
    missing = []
    for name, path in MODELS.items():
        if not path.exists():
            missing.append(str(path))
    if missing:
        print("❌  Model file(s) not found:")
        for m in missing:
            print(f"    {m}")
        print("\n📂  Please ensure your folder looks like:")
        print("    model_parameters/")
        print("    ├── fault-seg.pt")
        print("    └── laser.pt")
        sys.exit(1)
    print("✅  Both model files found.\n")


def compute_gflops_and_params(model):
    """Use ultralytics built-in info() for GFLOPs and param count."""
    try:
        # model.info() returns (layers, parameters, gradients, GFLOPs)
        result = model.info(verbose=False, imgsz=IMG_SIZE)
        if isinstance(result, (list, tuple)) and len(result) >= 4:
            n_layers, n_params, n_grads, gflops = result[:4]
            return float(gflops), int(n_params), int(n_layers)
    except Exception:
        pass

    # Fallback: manual thop profiling
    try:
        from thop import profile, clever_format
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        nn_model = model.model.to(DEVICE).eval()
        with torch.no_grad():
            macs, params = profile(nn_model, inputs=(dummy,), verbose=False)
        gflops = macs * 2 / 1e9
        return gflops, int(params), None
    except Exception as e:
        print(f"  [WARN] GFLOPs fallback failed: {e}")
        return None, None, None


def measure_latency(model):
    """
    Measure inference latency (ms) on CPU/GPU.
    Returns: mean_ms, std_ms, min_ms, max_ms
    """
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    nn_model = model.model.to(DEVICE).eval()

    # Warm-up
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = nn_model(dummy)

    # Timed runs
    times = []
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(TIMING_RUNS):
            if DEVICE == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = nn_model(dummy)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = nn_model(dummy)
                times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms":   round(float(times.mean()), 3),
        "std_ms":    round(float(times.std()),  3),
        "min_ms":    round(float(times.min()),  3),
        "max_ms":    round(float(times.max()),  3),
        "fps":       round(1000.0 / float(times.mean()), 1),
    }


def run_validation(model_path, yaml_path, model_label):
    """
    Run model.val() on the test split to get Precision, Recall, mAP50, mAP50-95.
    Returns dict of metric values (or None if dataset not available).
    """
    if not Path(yaml_path).exists():
        print(f"  [SKIP] {yaml_path} not found — skipping val metrics.")
        return None

    print(f"  Running validation on {yaml_path} ...")
    model = YOLO(str(model_path))
    try:
        results = model.val(
            data=yaml_path,
            imgsz=IMG_SIZE,
            split="test",       # use test split
            verbose=False,
            plots=False,
            save=False,
        )
        metrics = results.results_dict

        # Segmentation models return both box and mask metrics
        # Pull whichever key names exist
        def get(d, *keys):
            for k in keys:
                if k in d:
                    return round(float(d[k]), 4)
            return None

        box_p   = get(metrics, "metrics/precision(B)")
        box_r   = get(metrics, "metrics/recall(B)")
        box_m50 = get(metrics, "metrics/mAP50(B)")
        box_m95 = get(metrics, "metrics/mAP50-95(B)")

        seg_p   = get(metrics, "metrics/precision(M)")
        seg_r   = get(metrics, "metrics/recall(M)")
        seg_m50 = get(metrics, "metrics/mAP50(M)")
        seg_m95 = get(metrics, "metrics/mAP50-95(M)")

        # "Accuracy" for segmentation = mAP50-95 (mask) as overall score
        # We also report top-1 accuracy if classification head exists
        accuracy = seg_m95 if seg_m95 is not None else box_m95

        return {
            "box": {
                "precision":   box_p,
                "recall":      box_r,
                "mAP50":       box_m50,
                "mAP50_95":    box_m95,
            },
            "mask": {
                "precision":   seg_p,
                "recall":      seg_r,
                "mAP50":       seg_m50,
                "mAP50_95":    seg_m95,
            },
            "accuracy_proxy": accuracy,   # mAP50-95(M) as overall accuracy
            "raw": {k: round(float(v), 4) for k, v in metrics.items() if isinstance(v, (int, float))},
        }
    except Exception as e:
        print(f"  [ERROR] Validation failed: {e}")
        return None


def extract_all_metrics(name, model_path, yaml_path):
    """Full pipeline for one model."""
    sep = "─" * 55
    print(f"\n{'='*65}")
    print(f"  MODEL : {name.upper()}  ({model_path})")
    print(f"{'='*65}")

    results = {"model": name, "path": str(model_path)}

    # 1. Load
    print(f"\n[1/4] Loading model …")
    model = YOLO(str(model_path))

    # 2. Architecture stats
    print(f"[2/4] Computing GFLOPs & parameters …")
    gflops, n_params, n_layers = compute_gflops_and_params(model)
    model_size_mb = model_path.stat().st_size / 1e6
    results["architecture"] = {
        "GFLOPs":        gflops,
        "parameters_M":  round(n_params / 1e6, 3) if n_params else None,
        "parameters_raw": n_params,
        "layers":        n_layers,
        "model_size_MB": round(model_size_mb, 2),
    }
    print(f"  GFLOPs        : {gflops}")
    print(f"  Parameters    : {n_params:,} ({round(n_params/1e6,3)}M)" if n_params else "  Parameters    : N/A")
    print(f"  Layers        : {n_layers}")
    print(f"  Model size    : {model_size_mb:.2f} MB")

    # 3. Latency
    print(f"[3/4] Measuring latency ({TIMING_RUNS} runs on {DEVICE.upper()}) …")
    lat = measure_latency(model)
    results["latency"] = lat
    print(f"  Mean latency  : {lat['mean_ms']} ms  ±  {lat['std_ms']} ms")
    print(f"  Min / Max     : {lat['min_ms']} ms  /  {lat['max_ms']} ms")
    print(f"  FPS           : {lat['fps']}")

    # 4. Validation metrics
    print(f"[4/4] Running validation …")
    val = run_validation(model_path, yaml_path, name)
    results["validation"] = val

    if val:
        print(f"\n  ── Box Metrics ──────────────────────────────")
        print(f"  Precision     : {val['box']['precision']}")
        print(f"  Recall        : {val['box']['recall']}")
        print(f"  mAP@50        : {val['box']['mAP50']}")
        print(f"  mAP@50-95     : {val['box']['mAP50_95']}")
        print(f"\n  ── Mask (Segmentation) Metrics ──────────────")
        print(f"  Precision     : {val['mask']['precision']}")
        print(f"  Recall        : {val['mask']['recall']}")
        print(f"  mAP@50        : {val['mask']['mAP50']}")
        print(f"  mAP@50-95     : {val['mask']['mAP50_95']}")
        print(f"\n  ── Overall Accuracy Proxy ────────────────────")
        print(f"  Accuracy (mAP50-95 mask): {val['accuracy_proxy']}")
    else:
        print("  Validation skipped (dataset not available).")
        print("  → Metrics require the dataset YAML and images to be present.")
        print(f"  → Point yaml 'path:' to your dataset root and re-run.")

    return results


def print_summary_table(all_results):
    """Print a clean side-by-side comparison table."""
    print(f"\n\n{'='*65}")
    print(f"  FINAL SUMMARY  —  Side-by-Side Comparison")
    print(f"{'='*65}")

    header = f"{'Metric':<30} {'fault-seg':>16} {'laser':>16}"
    print(header)
    print("─" * 65)

    def row(label, key_chain, fmt="{}", suffix=""):
        vals = []
        for r in all_results:
            d = r
            try:
                for k in key_chain:
                    d = d[k]
                vals.append(fmt.format(d) + suffix if d is not None else "N/A")
            except (KeyError, TypeError):
                vals.append("N/A")
        print(f"  {label:<28} {vals[0]:>16} {vals[1]:>16}")

    print("  ARCHITECTURE")
    row("GFLOPs",               ["architecture","GFLOPs"],         "{:.1f}")
    row("Parameters (M)",       ["architecture","parameters_M"],   "{:.3f}", "M")
    row("Layers",               ["architecture","layers"],         "{}")
    row("Model Size (MB)",      ["architecture","model_size_MB"],  "{:.2f}", " MB")

    print("  LATENCY")
    row("Mean Latency",         ["latency","mean_ms"],             "{:.3f}", " ms")
    row("Std Dev",              ["latency","std_ms"],              "{:.3f}", " ms")
    row("Min Latency",          ["latency","min_ms"],              "{:.3f}", " ms")
    row("Max Latency",          ["latency","max_ms"],              "{:.3f}", " ms")
    row("FPS",                  ["latency","fps"],                 "{:.1f}")

    if any(r.get("validation") for r in all_results):
        print("  BOX METRICS")
        row("Precision (Box)",      ["validation","box","precision"],   "{:.4f}")
        row("Recall (Box)",         ["validation","box","recall"],      "{:.4f}")
        row("mAP@50 (Box)",         ["validation","box","mAP50"],       "{:.4f}")
        row("mAP@50-95 (Box)",      ["validation","box","mAP50_95"],    "{:.4f}")

        print("  SEGMENTATION MASK METRICS")
        row("Precision (Mask)",     ["validation","mask","precision"],   "{:.4f}")
        row("Recall (Mask)",        ["validation","mask","recall"],      "{:.4f}")
        row("mAP@50 (Mask)",        ["validation","mask","mAP50"],       "{:.4f}")
        row("mAP@50-95 (Mask)",     ["validation","mask","mAP50_95"],    "{:.4f}")

        print("  OVERALL")
        row("Accuracy (mAP50-95 M)",["validation","accuracy_proxy"],    "{:.4f}")

    print("─" * 65)


def save_results(all_results):
    """Save results to JSON."""
    out_path = Path("model_metrics_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅  Full results saved to: {out_path.resolve()}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    check_model_files()

    yaml_map = {
        "fault-seg": "fault_seg.yaml",
        "laser":     "laser.yaml",
    }

    all_results = []
    for name, model_path in MODELS.items():
        r = extract_all_metrics(name, model_path, yaml_map[name])
        all_results.append(r)

    print_summary_table(all_results)
    save_results(all_results)

    print("\n✅  Done! Check model_metrics_results.json for the full output.\n")
