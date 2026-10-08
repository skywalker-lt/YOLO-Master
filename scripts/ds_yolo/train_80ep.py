"""Resumable 80-epoch run with periodic backups (H200 launcher).

Rules: cache=disk under the pod's local /data; a copy of weights/last.pt, weights/best.pt, results.csv and args.yaml goes to
/training_data/<project>/<name>/ every 10 epochs and at the end; on launch the run resumes from the local last.pt, else from
the backup on /training_data, else starts fresh.
usage: python train_run.py --model <yaml or .pt> --name <run> [--pretrained <ckpt>] [--epochs 80] [--extra k=v ...]"""

import argparse
import shutil
import time
from pathlib import Path

import yaml

from ultralytics import YOLO

LOCAL = Path("/data/runs80")
BACKUP = Path("/training_data/runs80")
RECIPE = str(Path(__file__).with_name("recipe_yolo26m_stage2.yaml"))


def backup_run(run_dir: Path, name: str, tag: str):
    dst = BACKUP / name
    tmp = BACKUP / f".{name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "weights").mkdir(parents=True)
    for rel in ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"):
        src = run_dir / rel
        if src.exists():
            shutil.copy2(src, tmp / rel)
    (tmp / "BACKUP_INFO").write_text(f"{tag} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    old = BACKUP / f".{name}.old"
    shutil.rmtree(old, ignore_errors=True)
    if dst.exists():
        dst.rename(old)
    tmp.rename(dst)
    shutil.rmtree(old, ignore_errors=True)
    print(f"BACKUP {name}: {tag} -> {dst}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    run_dir = LOCAL / a.name
    local_last = run_dir / "weights" / "last.pt"
    backup_last = BACKUP / a.name / "weights" / "last.pt"
    if (
        not local_last.exists() and backup_last.exists()
    ):  # restore the backup into the local run dir, then resume from it
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.copytree(BACKUP / a.name, run_dir)
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
    for kv in a.extra:
        k, v = kv.split("=", 1)
        args[k] = yaml.safe_load(v)

    def on_fit_epoch_end(trainer):
        if (trainer.epoch + 1) % a.every == 0:
            backup_run(Path(trainer.save_dir), a.name, f"epoch {trainer.epoch + 1}")

    def on_train_end(trainer):
        backup_run(Path(trainer.save_dir), a.name, "final")

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
    model.train(**train_args)
