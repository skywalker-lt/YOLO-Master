#!/usr/bin/env python3
"""Model-only inference latency on the COCO val set, for N checkpoints side by side.

Times the FORWARD PASS AND NOTHING ELSE. Letterboxing, host->device transfer, decode and
NMS all happen outside the timed region: the images are preprocessed once, staged on the
GPU as a single tensor, and the timed loop does `net(x)` on a slice of it. What comes out
is the number you can put next to GFLOPs, not an end-to-end pipeline number.

Why real COCO images and not torch.randn
----------------------------------------
Every model here routes: v0.1-N (ModularRouterExpertMoE), UoMoE-N (UltraOptimizedMoE) and
EsMoE-N (ES_MOE) all pick experts from the input. Random noise drives the router to an
unrepresentative expert mix, which is the same defect that makes ultralytics' get_flops
non-reproducible on these models (it profiles a torch.empty). v0.1 and UoMoE hold top_k=2
regardless, so for them the mix changes but the FLOPs do not; ES_MOE additionally prunes
per image against dynamic_threshold, so the amount of compute genuinely varies. Feeding
real val images makes every one of those decisions representative.

Method
------
  * letterbox to a square imgsz (auto=False) so every tensor is identical and batching is legal
  * stage all --n-imgs images on the GPU up front, fp16 or fp32, matching --half
  * model.fuse() + .eval(), inference_mode, cudnn.benchmark on (warmup absorbs autotune)
  * --warmup iterations discarded, then --iters timed with CUDA events and a sync per iter
  * report mean/median/std/min/p90/p99 and FPS; p99 is the one that exposes routing jitter

Measure on an OTHERWISE-IDLE GPU -- never the one training. Compare models under identical
flags (same GPU, imgsz, batch, precision, warmup, iters) or the numbers mean nothing.

Examples
--------
    # three models, one table, COCO val, 640px fp16
    python scripts/reproduce/bench_coco_latency.py --imgsz 640 --half \
        --models "v0.1-N=runs/.../v0.1-N/weights/best.pt" \
                 "UoMoE-N=runs/.../UoMoE-N/weights/best.pt" \
                 "EsMoE-N=runs/.../EsMoE-N/weights/best.pt"

    # architecture only, no checkpoints -- YAML paths build an untrained net at the same nc
    python scripts/reproduce/bench_coco_latency.py --imgsz 640 \
        --models "v0.1-N=ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml" \
                 "UoMoE-N=ultralytics/cfg/models/master/v0_1/det/yolo-master-n-uomoe.yaml"

    # throughput instead of latency
    python scripts/reproduce/bench_coco_latency.py --batch 32 --iters 100 --models ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _stats(ms: np.ndarray) -> dict:
    return {
        "mean": float(ms.mean()), "median": float(np.median(ms)), "std": float(ms.std()),
        "min": float(ms.min()), "p90": float(np.percentile(ms, 90)), "p99": float(np.percentile(ms, 99)),
    }


def _resolve_val_images(data_yaml: str, split: str, n: int) -> list[Path]:
    """COCO val2017 paths. Handles both the txt-manifest layout (coco.yaml ships
    val2017.txt) and a plain image directory."""
    import yaml
    from ultralytics.utils.checks import check_file

    d = yaml.safe_load(open(check_file(data_yaml)))
    root = Path(d.get("path", "."))
    if not root.is_absolute():
        from ultralytics.utils import SETTINGS
        root = Path(SETTINGS["datasets_dir"]) / root
    entry = root / d[split]

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if entry.is_file():  # manifest: one relative image path per line
        paths = []
        for line in entry.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if p.is_absolute():
                paths.append(p)
            else:
                # coco.yaml's val2017.txt lines look like "./images/val2017/000000000139.jpg".
                # Strip the leading "./" explicitly -- lstrip("./") is a character-set strip
                # and would eat a leading directory name that happens to start with a dot.
                rel = line[2:] if line.startswith("./") else line
                paths.append((root / rel).resolve())
    elif entry.is_dir():
        paths = sorted(p for p in entry.rglob("*") if p.suffix.lower() in exts)
    else:
        raise FileNotFoundError(f"{entry} is neither a file nor a directory (from {data_yaml}:{split})")

    paths = [p for p in paths if p.exists()][:n]
    if not paths:
        raise FileNotFoundError(f"no readable images resolved from {entry}")
    return paths


def _stage_on_gpu(paths, imgsz, dev, dtype):
    """Letterbox -> RGB -> /255 -> NCHW, all of it BEFORE timing, resident on the GPU."""
    import cv2
    import torch
    from ultralytics.data.augment import LetterBox

    lb = LetterBox(new_shape=(imgsz, imgsz), auto=False, scaleup=True)
    out = torch.empty((len(paths), 3, imgsz, imgsz), device=dev, dtype=dtype)
    kept = 0
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = lb(image=im)[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        t = torch.from_numpy(np.ascontiguousarray(im)).to(dev, non_blocking=True)
        out[kept] = t.to(dtype).div_(255.0)
        kept += 1
    if kept == 0:
        raise RuntimeError("every image failed to decode")
    return out[:kept]


def _bench_one(label, weights, imgs, a, dev):
    import torch
    from ultralytics import YOLO

    ym = YOLO(str(weights))
    net = ym.model.to(dev).eval()
    net.fuse()
    if a.half:
        net.half()

    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    nc = getattr(net, "nc", None)

    # Deterministic GFLOPs on a REAL image (not ultralytics get_flops, which profiles a
    # torch.empty -- garbage values fire a different expert mix on every call).
    gflops = float("nan")
    try:
        import contextlib
        import io
        from copy import deepcopy

        import thop
        probe = imgs[:1].float().cpu()
        with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
            gflops = thop.profile(deepcopy(net).float().cpu().eval(), inputs=[probe], verbose=False)[0] / 1e9 * 2
    except Exception:  # noqa: BLE001
        pass

    n_avail = imgs.shape[0]
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for i in range(a.warmup):                      # warmup: cudnn autotune + clocks
            j = (i * a.batch) % max(1, n_avail - a.batch + 1)
            net(imgs[j:j + a.batch])
        torch.cuda.synchronize()

        t = np.empty(a.iters, dtype=np.float64)
        for i in range(a.iters):                       # cycle through the real val images
            j = (i * a.batch) % max(1, n_avail - a.batch + 1)
            x = imgs[j:j + a.batch]
            s.record()
            net(x)                                     # <-- the only thing timed
            e.record()
            torch.cuda.synchronize()
            t[i] = s.elapsed_time(e)

    st = _stats(t)
    return {"model": label, "weights": str(weights), "nc": nc, "params_M": round(n_params, 3),
            "gflops": round(gflops, 2), **{k: round(v, 4) for k, v in st.items()},
            "fps": round(1000.0 * a.batch / st["mean"], 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Model-only forward latency over COCO val.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help='One or more "LABEL=path" (path = .pt checkpoint or .yaml config). '
                         'A bare path is allowed; the filename becomes the label.')
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1, help="1 = latency; larger = throughput.")
    ap.add_argument("--half", action="store_true", help="FP16 (state the precision in results).")
    ap.add_argument("--n-imgs", type=int, default=512,
                    help="Real val images staged on the GPU and cycled through. "
                         "512 @ 640px fp16 is ~1.2 GB.")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--device", default="0")
    ap.add_argument("--json-out", default=None, help="Write the result rows to this path.")
    a = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("[bench] ERROR: no CUDA device. This benchmark is GPU-only "
              "(CUDA-event timing); run it on an idle GPU box.", file=sys.stderr)
        return 2

    dev = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(dev)
    torch.backends.cudnn.benchmark = True
    dtype = torch.half if a.half else torch.float
    prec = "fp16" if a.half else "fp32"
    gpu = torch.cuda.get_device_name(dev)

    paths = _resolve_val_images(a.data, a.split, a.n_imgs)
    imgs = _stage_on_gpu(paths, a.imgsz, dev, dtype)
    print(f"[bench] GPU={gpu}  imgsz={a.imgsz}  batch={a.batch}  {prec}  "
          f"warmup={a.warmup}  iters={a.iters}")
    print(f"[bench] staged {imgs.shape[0]} real {a.data}:{a.split} images on device "
          f"({imgs.element_size() * imgs.nelement() / 1e9:.2f} GB) -- preprocessing is OUTSIDE the timer\n")

    rows = []
    for spec in a.models:
        label, _, path = spec.partition("=")
        if not path:
            path, label = label, Path(label).stem
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            print(f"[skip] {label}: {p} not found", file=sys.stderr)
            continue
        r = _bench_one(label, p, imgs, a, dev)
        r.update(gpu=gpu, precision=prec, imgsz=a.imgsz, batch=a.batch, n_imgs=int(imgs.shape[0]))
        rows.append(r)
        print(f"[{r['model']}] params={r['params_M']}M  GFLOPs={r['gflops']}  nc={r['nc']}\n"
              f"    ms  mean={r['mean']:.3f}  median={r['median']:.3f}  std={r['std']:.3f}  "
              f"min={r['min']:.3f}  p90={r['p90']:.3f}  p99={r['p99']:.3f}\n"
              f"    {r['fps']:.1f} img/s\n")

    if rows:
        w = max(len(r["model"]) for r in rows)
        print(f"\n{'model'.ljust(w)} | params M | GFLOPs | mean ms | median ms | p99 ms |   FPS")
        print(f"{'-' * w}-+----------+--------+---------+-----------+--------+------")
        for r in rows:
            print(f"{r['model'].ljust(w)} | {r['params_M']:>8.3f} | {r['gflops']:>6.2f} | "
                  f"{r['mean']:>7.3f} | {r['median']:>9.3f} | {r['p99']:>6.3f} | {r['fps']:>5.1f}")
        base = rows[0]
        print(f"\nrelative to {base['model']} (mean ms):")
        for r in rows[1:]:
            print(f"  {r['model']}: {r['mean'] / base['mean']:.3f}x")

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\n[bench] wrote {a.json_out}")
    print("RESULT_JSON " + json.dumps(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
