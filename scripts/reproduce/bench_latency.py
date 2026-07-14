#!/usr/bin/env python3
"""Rigorous inference-latency benchmark for a YOLO-Master checkpoint.

Always runs the clean model-only pass; add --e2e to ALSO run the end-to-end pass:
  model-only  — pure fused forward on a fixed dummy input, timed with CUDA events
                (warmup + many iters) -> mean/median/std/p90/p99 (ms) + FPS. The clean,
                reproducible, cross-paper-comparable number.
  --e2e       — full pipeline (letterbox preprocess + forward + NMS) over REAL images,
                averaging ultralytics' per-stage speed. Real-world, but data-dependent
                (NMS scales with #detections -- big on dense AI-TOD).

Measure on an OTHERWISE-IDLE GPU (never the one training). For max reproducibility lock the
clocks first:  sudo nvidia-smi -lgc <graphics_clock>  (or at least confirm steady state).
Compare models under IDENTICAL flags (same GPU, imgsz, batch, precision, warmup, iters).

Examples:
    python scripts/reproduce/bench_latency.py --weights best.pt --imgsz 800 --half
    python scripts/reproduce/bench_latency.py --weights best.pt --imgsz 800 --half --e2e --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _stats(ms: np.ndarray) -> str:
    return (f"mean={ms.mean():.3f}  median={np.median(ms):.3f}  std={ms.std():.3f}  "
            f"min={ms.min():.3f}  p90={np.percentile(ms, 90):.3f}  p99={np.percentile(ms, 99):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Rigorous latency benchmark.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=1, help="1 = latency; large = throughput.")
    ap.add_argument("--half", action="store_true", help="FP16 (state the precision in results).")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--device", default="0")
    # end-to-end (additional) pass
    ap.add_argument("--e2e", action="store_true", help="ALSO time preprocess+forward+NMS over real images.")
    ap.add_argument("--source", default=None, help="e2e image dir (default: <data path>/images/<split>).")
    ap.add_argument("--data", default="ultralytics/cfg/datasets/AI-TOD-v2.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--max-det", type=int, default=1500)
    ap.add_argument("--e2e-imgs", type=int, default=500, help="How many real images to time in --e2e.")
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO

    dev = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(dev)
    torch.backends.cudnn.benchmark = True  # autotune for the fixed input size (warmup absorbs it)

    ym = YOLO(a.weights)
    n_params = sum(p.numel() for p in ym.model.parameters()) / 1e6
    # Deterministic full-res GFLOPs (fixed-seed input). NOT ultralytics get_flops:
    # it profiles a torch.empty 32x32 -> input-dependent MoE routing makes it
    # non-reproducible (garbage values fire different experts) once x625'd.
    try:
        import contextlib
        import io
        from copy import deepcopy
        import thop
        _g = torch.Generator().manual_seed(0)
        _xf = torch.randn(1, 3, a.imgsz, a.imgsz, generator=_g)
        with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
            gflops = thop.profile(deepcopy(ym.model).eval(), inputs=[_xf], verbose=False)[0] / 1e9 * 2
    except Exception:  # noqa: BLE001
        gflops = float("nan")
    gpu_name = torch.cuda.get_device_name(dev)
    prec = "fp16" if a.half else "fp32"
    print(f"[bench] {Path(a.weights).name}  GPU={gpu_name}  imgsz={a.imgsz}  batch={a.batch}  {prec}")
    print(f"[bench] params={n_params:.3f}M  GFLOPs={gflops:.1f} @ {a.imgsz}px")

    # ---- model-only: fused forward on a fixed input, CUDA-event timed (always) ----
    net = ym.model.to(dev).eval()
    net.fuse()
    if a.half:
        net.half()
    dtype = torch.half if a.half else torch.float
    x = torch.randn(a.batch, 3, a.imgsz, a.imgsz, device=dev, dtype=dtype)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for _ in range(a.warmup):
            net(x)
        torch.cuda.synchronize()
        t = np.empty(a.iters, dtype=np.float64)
        for i in range(a.iters):
            s.record()
            net(x)
            e.record()
            torch.cuda.synchronize()
            t[i] = s.elapsed_time(e)  # ms
    print(f"[bench] MODEL-ONLY  (warmup={a.warmup} iters={a.iters})")
    print(f"  latency ms: {_stats(t)}")
    print(f"  throughput: {1000.0 * a.batch / t.mean():.1f} img/s  (batch={a.batch})")
    print("RESULT_JSON " + json.dumps({"mode": "model-only", "params_M": round(n_params, 3),
                                       "gflops": round(gflops, 2), "imgsz": a.imgsz, "batch": a.batch,
                                       "precision": prec, "gpu": gpu_name,
                                       "ms_mean": round(float(t.mean()), 4), "ms_median": round(float(np.median(t)), 4),
                                       "ms_p90": round(float(np.percentile(t, 90)), 4),
                                       "ms_p99": round(float(np.percentile(t, 99)), 4),
                                       "fps": round(1000.0 * a.batch / float(t.mean()), 1)}))

    if not a.e2e:
        return 0

    # ---- end-to-end: preprocess + forward + NMS over REAL images (data-dependent) ----
    import yaml
    from ultralytics.utils.checks import check_file
    if a.source:
        img_dir = Path(a.source)
    else:
        d = yaml.safe_load(open(check_file(a.data)))
        img_dir = Path(d["path"]) / d[a.split]
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)[: a.e2e_imgs + a.warmup]
    assert imgs, f"no images in {img_dir}"
    pre, inf, post = [], [], []
    seen = 0
    for r in ym.predict(source=[str(p) for p in imgs], imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                        max_det=a.max_det, half=a.half, device=a.device, batch=a.batch,
                        stream=True, verbose=False, save=False):
        seen += 1
        if seen <= a.warmup:      # discard warmup images
            continue
        pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
    pre, inf, post = np.array(pre), np.array(inf), np.array(post)
    tot = pre + inf + post
    print(f"[bench] END-TO-END  over {len(tot)} real images from {img_dir.name}  (conf={a.conf} max_det={a.max_det})")
    print(f"  preprocess ms:  {_stats(pre)}")
    print(f"  inference  ms:  {_stats(inf)}")
    print(f"  postproc/NMS ms:{_stats(post)}   <- data-dependent (scales with #detections)")
    print(f"  TOTAL      ms:  {_stats(tot)}")
    print(f"  throughput: {1000.0 / tot.mean():.1f} img/s end-to-end")
    print("RESULT_JSON " + json.dumps({"mode": "e2e", "params_M": round(n_params, 3), "gflops": round(gflops, 2),
                                       "imgsz": a.imgsz, "precision": prec, "gpu": gpu_name, "n_imgs": len(tot),
                                       "pre_ms": round(float(pre.mean()), 4), "inf_ms": round(float(inf.mean()), 4),
                                       "nms_ms": round(float(post.mean()), 4), "total_ms": round(float(tot.mean()), 4),
                                       "total_p99_ms": round(float(np.percentile(tot, 99)), 4),
                                       "fps": round(1000.0 / float(tot.mean()), 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
