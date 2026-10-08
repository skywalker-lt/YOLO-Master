"""Resumable 80-epoch run with periodic backups (H200 launcher).

Rules: cache=disk under the pod's local /data; a copy of weights/last.pt, weights/best.pt, results.csv and args.yaml goes to
/training_data/<project>/<name>/ every 10 epochs and at the end; on launch the run resumes from the local last.pt, else from
the backup on /training_data, else starts fresh. --mem-fraction caps this process's share of GPU memory so that several
runs can share one GPU without one of them failing when their validation phases coincide.
usage: python train_80ep.py --model <yaml or .pt> --name <run> [--pretrained <ckpt>] [--epochs 80] [--project runs80]
       [--scales 512,768] [--freeze-stem 6] [--mem-fraction 0.31] [--extra k=v ...]"""

import argparse
import os
import shutil
import time
from pathlib import Path

import yaml

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch is imported

from ultralytics import YOLO  # noqa: E402

LOCAL = Path("/data/runs80")  # replaced by --project
BACKUP = Path("/training_data/runs80")
RECIPE = str(Path(__file__).with_name("recipe_yolo26m_stage2.yaml"))


def backup_run(run_dir: Path, name: str, tag: str):
    """Copy the run's checkpoints and logs to the shared volume. File contents only (the volume refuses metadata
    changes), and never raises: a failed backup is logged, training continues."""
    try:
        dst = BACKUP / name
        tmp = BACKUP / f".{name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "weights").mkdir(parents=True)
        for rel in ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"):
            src = run_dir / rel
            if src.exists():
                shutil.copyfile(src, tmp / rel)
        (tmp / "BACKUP_INFO").write_text(f"{tag} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        old = BACKUP / f".{name}.old"
        shutil.rmtree(old, ignore_errors=True)
        if dst.exists():
            dst.rename(old)
        tmp.rename(dst)
        shutil.rmtree(old, ignore_errors=True)
        print(f"BACKUP {name}: {tag} -> {dst}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"BACKUP FAILED {name}: {tag}: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--project", default="runs80", help="run directory under /data (backups under /training_data)")
    ap.add_argument(
        "--scales", default=None, help="comma-separated sizes, e.g. 512,768: every batch is resized to one of them at random"
    )
    ap.add_argument("--freeze-stem", type=int, default=0, help="freeze layers 0..N-1 (weights and batch-norm stats)")
    ap.add_argument(
        "--mem-fraction", type=float, default=0.0, help="cap on this process's share of GPU 0 memory (0 = none)"
    )
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    LOCAL, BACKUP = Path("/data") / a.project, Path("/training_data") / a.project
    if a.mem_fraction > 0:
        import torch

        torch.cuda.set_per_process_memory_fraction(a.mem_fraction, 0)
        print(f"MEM FRACTION {a.mem_fraction}", flush=True)
    run_dir = LOCAL / a.name
    local_last = run_dir / "weights" / "last.pt"
    backup_last = BACKUP / a.name / "weights" / "last.pt"
    if (
        not local_last.exists() and backup_last.exists()
    ):  # restore the backup into the local run dir, then resume from it
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.copytree(BACKUP / a.name, run_dir, copy_function=shutil.copyfile)
        print(
            f"RESTORED {a.name} from {BACKUP / a.name} ({(BACKUP / a.name / 'BACKUP_INFO').read_text().strip()})",
            flush=True,
        )
    with open(RECIPE) as f:
        recipe = yaml.safe_load(f)
    recipe["weight_decay"] = recipe["weight_decay"] * 128 / a.batch
    args = dict(
        data="/data/datasets/coco/coco.yaml",
        epochs=a.epochs,
        batch=a.batch,
        imgsz=640,
        cache="disk",
        workers=a.workers,
        device=0,
        project=str(LOCAL),
        name=a.name,
        exist_ok=True,
        seed=0,
        plots=False,
        moe_num_experts=4,
        moe_top_k=2,
        **recipe,
    )
    if a.pretrained:
        args["pretrained"] = a.pretrained
    if a.freeze_stem:
        args["freeze"] = list(range(a.freeze_stem))
    for kv in a.extra:
        k, v = kv.split("=", 1)
        args[k] = yaml.safe_load(v)

    def on_fit_epoch_end(trainer):
        if (trainer.epoch + 1) % a.every == 0:
            backup_run(Path(trainer.save_dir), a.name, f"epoch {trainer.epoch + 1}")

    def on_train_end(trainer):
        backup_run(Path(trainer.save_dir), a.name, "final")

    def on_train_start(trainer):
        if a.scales:  # two-scale training: the same interpolation the trainer's multi_scale option uses, at fixed sizes
            import random

            from torch.nn import functional as F

            sizes = [int(v) for v in a.scales.split(",")]
            original = trainer.preprocess_batch

            def preprocess_batch(batch):
                batch = original(batch)
                sz = random.choice(sizes)
                if sz != batch["img"].shape[-1]:
                    batch["img"] = F.interpolate(batch["img"], size=(sz, sz), mode="bilinear", align_corners=False)
                return batch

            trainer.preprocess_batch = preprocess_batch
            print(f"TWO-SCALE {sizes}", flush=True)

    def on_train_batch_start(trainer):
        if a.freeze_stem:  # the trainer freezes the weights; the batch-norm statistics stay frozen too
            for layer in list(trainer.model.model)[: a.freeze_stem]:
                layer.eval()

    if local_last.exists():
        print(f"RESUME {a.name} from {local_last}", flush=True)
        model = YOLO(str(local_last))
        train_args = {"resume": True, "cache": "disk"}
    else:
        print("ARGS", {k: args[k] for k in sorted(args)}, flush=True)
        model = YOLO(a.model)
        train_args = args
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_start", on_train_batch_start)
    model.train(**train_args)
