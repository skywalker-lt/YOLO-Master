#!/usr/bin/env python3
"""Train the P2 tiny-object-head variants of YOLO-Master.

Detached from the merged reproduction scripts (scripts/reproduce/, upstream #81):
this module owns everything P2-specific and REUSES the reproduction training core
(train_one, callbacks, summary, arg parser) without modifying it.

P2 variants
-----------
  v0.1-N-P2       static  ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2.yaml
  v0.1-N-P2-PSA   static  ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2-psa.yaml
  EsMoE-N-P2      runtime  P2/4 head grafted onto the ES_MOE backbone (generated yaml)
  EsMoE-N-P2-PSA  runtime  the above + a C2PSA transformer head at P5

-P2 adds a stride-4 tiny-object detection head -> Detect(P2,P3,P4,P5); -P2-PSA also
adds a channel-preserving C2PSA transformer head at the low-resolution P5, following
the TPH-YOLOv5 paper's extra-head + transformer-head ideas. EsMoE variants keep their
ES_MOE blocks, so pass --no-sparse-eval for correct (dense) evaluation.

Usage
-----
  python scripts/p2/reproduce_p2.py --dataset visdrone --model EsMoE-N-P2 \
         --no-sparse-eval --imgsz 640 --epochs 300 --batch 32
  python scripts/p2/reproduce_p2.py --dataset sku110k --model all --check-build
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

# reuse the (clean, upstreamed) reproduction core
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduce"))
import yaml  # noqa: E402
from _reproduce_common import (  # noqa: E402
    ROOT, ModelSpec, DatasetSpec, build_parser, train_one, write_summary,
)

DATASETS = {
    "visdrone": DatasetSpec("VisDrone", "VisDrone.yaml", "runs/p2/visdrone"),
    "sku110k": DatasetSpec("SKU-110K", "SKU-110K.yaml", "runs/p2/sku110k"),
}
_ESMOE_BASE = "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml"
_GEN_CFG_DIR = Path(tempfile.gettempdir()) / "yolo_master_p2_cfgs"


def build_p2_head(transformer: bool) -> list:
    """The P2 detection head grafted onto the base EsMoE-N backbone.

    EsMoE-N backbone P-level feature sources (0-indexed layers):
        P2/4 = layer 3 (ES_MOE[256]),  P3/8 = 6,  P4/16 = 9,  P5/32 = 12.
    The FPN top-down path is carried one level further to stride-4 and the PAN
    bottom-up path is re-rooted at P2 -> Detect(P2, P3, P4, P5). With
    transformer=True a channel-preserving C2PSA head is appended at P5 (-P2-PSA).
    """
    head = [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],   # 13
        [[-1, 9], 1, "Concat", [1]],                     # 14  cat backbone P4
        [-1, 2, "C3k2", [512, True]],                    # 15  P4 td
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],   # 16
        [[-1, 6], 1, "Concat", [1]],                     # 17  cat backbone P3
        [-1, 2, "C3k2", [256, True]],                    # 18  P3 td
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],   # 19
        [[-1, 3], 1, "Concat", [1]],                     # 20  cat backbone P2
        [-1, 2, "C3k2", [128, True]],                    # 21  P2/4-tiny   -> Detect P2
        [-1, 1, "Conv", [128, 3, 2]],                    # 22
        [[-1, 18], 1, "Concat", [1]],                    # 23  cat head P3
        [-1, 2, "C3k2", [256, True]],                    # 24            -> Detect P3
        [-1, 1, "Conv", [256, 3, 2]],                    # 25
        [[-1, 15], 1, "Concat", [1]],                    # 26  cat head P4
        [-1, 2, "C3k2", [512, True]],                    # 27            -> Detect P4
        [-1, 1, "Conv", [512, 3, 2]],                    # 28
        [[-1, 12], 1, "Concat", [1]],                    # 29  cat backbone P5
        [-1, 2, "C3k2", [512, True]],                    # 30  P5/32-large
    ]
    if transformer:
        head.append([-1, 1, "C2PSA", [512]])             # 31  transformer head @ P5
        detect = [21, 24, 27, 31]
    else:
        detect = [21, 24, 27, 30]
    head.append([detect, 1, "Detect", ["nc"]])
    return head


def _generate_esmoe_cfg(name: str, transformer: bool) -> str:
    """Graft build_p2_head() onto the base EsMoE-N yaml, write a temp yaml, return its path."""
    cfg = yaml.safe_load((ROOT / _ESMOE_BASE).read_text())
    cfg["head"] = build_p2_head(transformer)
    _GEN_CFG_DIR.mkdir(parents=True, exist_ok=True)
    out = _GEN_CFG_DIR / f"{name.replace('.', '_')}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
    return str(out)  # absolute -> train_one's `ROOT / cfg` keeps it absolute


def p2_specs() -> list[ModelSpec]:
    """P2 variants as plain ModelSpecs with cfg RESOLVED to a path (static or generated)."""
    return [
        ModelSpec("v0.1-N-P2", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2.yaml", uses_esmoe=False),
        ModelSpec("v0.1-N-P2-PSA", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2-psa.yaml", uses_esmoe=False),
        ModelSpec("EsMoE-N-P2", _generate_esmoe_cfg("EsMoE-N-P2", False), uses_esmoe=True),
        ModelSpec("EsMoE-N-P2-PSA", _generate_esmoe_cfg("EsMoE-N-P2-PSA", True), uses_esmoe=True),
    ]


def main() -> int:
    # pick the dataset first, then reuse the reproduction parser with P2 --model choices
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", choices=list(DATASETS), default="visdrone")
    dataset = DATASETS[pre.parse_known_args()[0].dataset]

    specs = p2_specs()
    parser = build_parser(dataset)
    parser.add_argument("--dataset", choices=list(DATASETS), default="visdrone",
                        help="Which vertical dataset to train on.")
    for a in parser._actions:                 # repoint --model at the P2 registry
        if a.dest == "model":
            a.choices = [s.name for s in specs] + ["all"]
            a.default = "all"
            a.help = "P2 variant to train, or 'all' (default)."
    args = parser.parse_args()

    project = Path(args.project) if Path(args.project).is_absolute() else ROOT / args.project
    sel = specs if args.model == "all" else [s for s in specs if s.name == args.model]

    print(f"[p2:{dataset.name}] data={dataset.data}  project={project}  sparse_eval={args.sparse_eval}")
    for s in sel:
        note = "ES_MOE -> pass --no-sparse-eval for dense eval" if s.uses_esmoe else "no ES_MOE"
        print(f"  - {s.name:<14} cfg={s.cfg}  ({note})")

    if args.dry_run:
        return 0
    if args.check_build:
        from ultralytics.nn.tasks import DetectionModel
        for s in sel:
            m = DetectionModel(str(ROOT / s.cfg), ch=3, nc=80, verbose=False)
            print(f"[build-ok] {s.name}: {sum(p.numel() for p in m.parameters()) / 1e6:.3f}M")
        return 0
    if args.summary_only:
        print("[summary]", write_summary(project, dataset, sel, sparse_eval=args.sparse_eval))
        return 0

    project.mkdir(parents=True, exist_ok=True)
    statuses = []
    for s in sel:
        try:
            statuses.append(train_one(args, dataset, s, project))
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {s.name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            statuses.append({"model": s.name, "status": "failed", "error": str(exc)})
            if args.stop_on_failure:
                break
        finally:
            try:
                write_summary(project, dataset, sel, sparse_eval=args.sparse_eval)
            except OSError as e:
                print(f"[summary-warn] {e}", flush=True)

    print(f"\n[p2:{dataset.name}] DONE")
    for st in statuses:
        print("  ", st)
    return 0 if all(st.get("status") in {"ok", "resumed", "skipped"} for st in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
