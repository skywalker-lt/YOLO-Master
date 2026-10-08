"""Gate G2 of the seminar-4 two-scale receipt: mix 512 and 768 val2017 dumps per image with the thumbnail count router.

The router is the seminar's: a 5-fold out-of-fold ridge from the YOLO26-N@320 pooled stem feature (fixed, public weights) to
the log small-object count, images ranked by the prediction and the top share sent to 768. Prints, for the given pair of
COCO-format dumps: each scale alone, the learned router at the given shares, the ground-truth count rule, and the random
null (seed-averaged over three draws).
usage: python receipt_mix.py <dump512_coco.json> <dump768_coco.json> [--shares 0.43,0.52] [--feats val2017_stem_pooled.npz]"""

import argparse
import contextlib
import io
import json
import os

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("d512")
ap.add_argument("d768")
ap.add_argument("--shares", default="0.43,0.52")
ap.add_argument("--ann", default="/data/tmp/ds-yolo/seminar4/inputs/dumps/instances_val2017.json")
ap.add_argument("--feats", default="/data/tmp/ds-yolo/seminar4/inputs/dumps/val2017_stem_pooled.npz")
ap.add_argument("--tag", default="")
a = ap.parse_args()

gt = COCO(a.ann)
IMG = sorted(gt.getImgIds())
pos = {i: k for k, i in enumerate(IMG)}
rng = np.random.default_rng(0)
Z = np.load(a.feats)
order = np.array([pos[int(i)] for i in Z["image_id"]])
X = np.zeros((len(IMG), Z["n320"].shape[1]))
X[order] = Z["n320"]


def evaluate(res, tag):
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(res), "bbox")
        ev.params.imgIds = IMG
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    st = ev.stats
    print(f"MIX {a.tag}{tag:58s} AP={st[0]:.4f} S/M/L={st[3]:.4f}/{st[4]:.4f}/{st[5]:.4f}", flush=True)
    return st[0]


dets = {}
for k, f in (("512", a.d512), ("768", a.d768)):
    r = json.load(open(f))
    dets[k] = {}
    for d in r:
        dets[k].setdefault(d["image_id"], []).append(d)
    evaluate(r, f"{k} alone")


def mix(choice):
    out = []
    for i, c in zip(IMG, choice):
        out += dets["768" if c else "512"].get(i, [])
    return out


def cv_ridge(X, y, lam=100.0, folds=5):
    n = len(y)
    idx = rng.permutation(n)
    pred = np.zeros(n)
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = (X[tr] - mu) / sd
        Xe = (X[te] - mu) / sd
        w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ (y[tr] - y[tr].mean()))
        pred[te] = Xe @ w + y[tr].mean()
    return pred


def top(score, share):
    k = int(round(share * len(IMG)))
    ch = np.zeros(len(IMG), int)
    ch[np.argsort(-score)[:k]] = 1
    return ch


anns = {i: gt.loadAnns(gt.getAnnIds(imgIds=i, iscrowd=False)) for i in IMG}
nsmall = np.array([sum(ann["area"] < 32 * 32 for ann in anns[i]) for i in IMG], float)
p_cnt = cv_ridge(X, np.log1p(nsmall))
for share in (float(s) for s in a.shares.split(",")):
    evaluate(mix(top(p_cnt, share)), f"learned n320 -> small-object count, 768 share {share:.2f}")
    evaluate(mix(top(nsmall + 1e-3 * rng.random(len(IMG)), share)), f"GT small-object count rule, 768 share {share:.2f}")
    nulls = [evaluate(mix((rng.random(len(IMG)) < share).astype(int)), f"random null draw, 768 share {share:.2f}") for _ in range(3)]
    print(f"MIX {a.tag}random null mean of three draws, 768 share {share:.2f}: AP={np.mean(nulls):.4f}", flush=True)
print("MIX DONE")
