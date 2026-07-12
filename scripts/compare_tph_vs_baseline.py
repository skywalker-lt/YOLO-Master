#!/usr/bin/env python3
"""Comprehensive A/B test: YOLO-Master-v0.1-N (baseline) vs -TPH at imgsz=640.

Both checkpoints are trained on VisDrone at 640 for 300 epochs, so this is a
fair, budget-matched comparison isolating the extra P2/4 tiny-object head.

Reports, per model:
  * Accuracy  : mAP50, mAP50-95, mean P/R on the VisDrone val split (+ per-class
                mAP50-95, which surfaces WHERE the P2 head helps -- small classes).
  * Model     : params (M), GFLOPs @ imgsz.
  * Latency   : batch-1 eager forward (median/mean ms) and compute-bound
                per-frame throughput at a larger batch. NOTE: run this on an IDLE
                GPU -- latency under a concurrent training job is meaningless.

Usage:
    python scripts/compare_tph_vs_baseline.py                 # full A/B (val + latency)
    python scripts/compare_tph_vs_baseline.py --no-latency    # accuracy only (safe on busy GPU)
    python scripts/compare_tph_vs_baseline.py --no-val        # latency only (needs idle GPU)
    python scripts/compare_tph_vs_baseline.py --device 1      # pick an idle GPU
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]

# The budget-matched pair (both VisDrone / imgsz 640 / 300 epochs).
MODELS = {
    "v0.1-N (baseline)": ROOT / "runs/baseline/result-v0.1n-visdrone/weights/best.pt",
    "v0.1-N-TPH":        ROOT / "v0.1-N-TPH-640/VisDrone_v0.1-N-TPH/weights/best.pt",
}


def measure_latency(net, imgsz, device, half, bs, iters, warm):
    dt = torch.half if half else torch.float
    net = (net.half() if half else net.float()).to(device).eval()
    x = torch.randn(bs, 3, imgsz, imgsz, device=device, dtype=dt)
    times = []
    with torch.no_grad():
        for _ in range(warm):
            net(x)
        torch.cuda.synchronize(device)
        for _ in range(iters):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            net(x)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    mean = sum(times) / len(times)
    return {"bs": bs, "mean_ms": mean, "median_ms": times[len(times) // 2],
            "per_frame_ms": mean / bs, "fps": 1000.0 * bs / mean}


def evaluate(tag, pt, args):
    print(f"\n{'='*70}\n{tag}  <-  {pt}\n{'='*70}")
    assert Path(pt).exists(), f"checkpoint not found: {pt}"
    model = YOLO(str(pt))
    names = model.names
    out = {"tag": tag, "pt": str(pt)}

    # --- model stats ---
    net = model.model
    out["params_M"] = sum(p.numel() for p in net.parameters()) / 1e6
    try:
        out["gflops"] = get_flops(net, imgsz=args.imgsz)
    except Exception:
        out["gflops"] = float("nan")

    # --- accuracy (VisDrone val) ---
    if not args.no_val:
        m = model.val(data=args.data, imgsz=args.imgsz, batch=1, half=True,
                      workers=0, split="val", device=args.device, plots=False,
                      verbose=False)
        out["map50"] = float(m.box.map50)
        out["map"] = float(m.box.map)
        out["mp"] = float(m.box.mp)
        out["mr"] = float(m.box.mr)
        out["speed"] = dict(m.speed)                 # ms preprocess/inference/postprocess per image
        out["per_class_map"] = {names[i]: float(v) for i, v in enumerate(m.box.maps)}

    # --- latency (needs idle GPU) ---
    if not args.no_latency:
        dev = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
        out["lat_b1"] = measure_latency(model.model, args.imgsz, dev, half=True,
                                        bs=1, iters=args.iters, warm=args.warm)
        out["lat_batched"] = measure_latency(model.model, args.imgsz, dev, half=True,
                                             bs=args.batch, iters=args.iters, warm=args.warm)
    return out


def fmt_table(results, args):
    a, b = results  # baseline, tph
    L = []
    L.append("\n" + "#" * 78)
    L.append(f"# YOLO-Master  v0.1-N  vs  v0.1-N-TPH   (VisDrone val, imgsz={args.imgsz})")
    L.append("#" * 78)
    L.append(f"\n{'metric':<22}{a['tag']:>18}{b['tag']:>18}{'delta':>14}")
    L.append("-" * 72)

    def row(label, ka, kb, pct=False, better_up=True, fmt="{:.3f}"):
        va, vb = ka, kb
        d = vb - va
        arrow = "" if abs(d) < 1e-9 else ("↑" if (d > 0) == better_up else "↓")
        ds = (fmt.format(d) if not pct else f"{d:+.3f}") + f" {arrow}"
        L.append(f"{label:<22}{fmt.format(va):>18}{fmt.format(vb):>18}{ds:>14}")

    row("params (M)", a["params_M"], b["params_M"], better_up=False)
    row("GFLOPs", a["gflops"], b["gflops"], better_up=False, fmt="{:.1f}")
    if "map50" in a:
        row("mAP50", a["map50"], b["map50"])
        row("mAP50-95", a["map"], b["map"])
        row("precision", a["mp"], b["mp"])
        row("recall", a["mr"], b["mr"])
        L.append(f"{'val speed inf(ms/img)':<22}{a['speed']['inference']:>18.2f}{b['speed']['inference']:>18.2f}{'':>14}")
    if "lat_b1" in a:
        L.append("-" * 72)
        L.append(f"{'latency b=1 (ms)':<22}{a['lat_b1']['median_ms']:>18.3f}{b['lat_b1']['median_ms']:>18.3f}")
        L.append(f"{'batched ms/frame':<22}{a['lat_batched']['per_frame_ms']:>18.3f}{b['lat_batched']['per_frame_ms']:>18.3f}")
        L.append(f"{'batched FPS':<22}{a['lat_batched']['fps']:>18.1f}{b['lat_batched']['fps']:>18.1f}")

    # per-class: where does TPH help?
    if "per_class_map" in a:
        L.append("\nPer-class mAP50-95 (sorted by TPH gain — small classes should top):")
        L.append(f"{'class':<18}{'baseline':>12}{'TPH':>12}{'delta':>10}")
        L.append("-" * 52)
        deltas = sorted(((c, a["per_class_map"][c], b["per_class_map"][c],
                          b["per_class_map"][c] - a["per_class_map"][c])
                         for c in a["per_class_map"]), key=lambda r: -r[3])
        for c, va, vb, d in deltas:
            L.append(f"{c:<18}{va:>12.3f}{vb:>12.3f}{d:>+10.3f}")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="VisDrone.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=32, help="batch for compute-bound throughput")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warm", type=int, default=20)
    p.add_argument("--no-val", action="store_true", help="skip accuracy (latency only)")
    p.add_argument("--no-latency", action="store_true", help="skip latency (safe on a busy GPU)")
    p.add_argument("--out", default=str(ROOT / "scripts/reproduce/results/tph_vs_baseline_640.md"))
    args = p.parse_args()

    results = [evaluate(tag, pt, args) for tag, pt in MODELS.items()]
    table = fmt_table(results, args)
    print(table)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(table)
    print(f"\n[saved] {outp}")


if __name__ == "__main__":
    main()
