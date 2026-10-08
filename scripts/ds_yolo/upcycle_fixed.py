"""Upcycle the Objects365 YOLO26-M into the two routed arms of the 80-epoch test (H200 tool).
Experts start as exact copies of the dense kernels (eps 0); one linear router is set to a balanced nearest-centroid
partition (k-means on the pooled layer-5 feature of 20,000 training images) and frozen. Two checkpoints:
hard top-1 (four static branches) and hard top-2 (six pair branches)."""

import copy
import glob
import os
import random
import time

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules.moe.weight_bank import bank_modules, convert_to_weight_bank, fit_fixed_router

SRC = "/data/weights/yolo26m-objv1-150.pt"
OUT = "/data/weights"
os.makedirs(OUT, exist_ok=True)


class Imgs(Dataset):
    def __init__(self, files, flip=False):
        self.files, self.flip, self.lb = files, flip, LetterBox((640, 640), auto=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = self.lb(image=cv2.imread(self.files[i]))
        if self.flip:
            im = im[:, ::-1]
        return torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy())


@torch.no_grad()
def pooled_features(stem, files, flip=False):
    out = []
    for x in DataLoader(Imgs(files, flip), batch_size=128, num_workers=32):
        x = x.cuda().float() / 255
        for layer in stem:
            x = layer(x)
        out.append(x.float().mean((2, 3)).cpu())
    return torch.cat(out)


if __name__ == "__main__":
    ck = torch.load(SRC, map_location="cpu", weights_only=False)
    model = (ck.get("ema") or ck["model"]).float().eval()
    state = convert_to_weight_bank(model.model, 6, 22, experts=4, top_k=1, hidden=0, eps=0.0, hard=True)
    router = bank_modules(model)[0].router
    model.cuda()
    random.seed(0)
    train = sorted(glob.glob("/data/datasets/coco/images/train2017/*.jpg"))
    random.shuffle(train)
    train = train[:20000]
    val = sorted(glob.glob("/data/datasets/coco/images/val2017/*.jpg"))
    stem = list(model.model)[:6]
    t0 = time.time()
    feats = pooled_features(stem, train).cuda()
    before, after = fit_fixed_router(router, feats)
    print(
        f"ROUTER fit on {len(feats)} train images in {time.time() - t0:.0f} s; shares before balancing {[round(float(v), 3) for v in before]}, after {[round(float(v), 3) for v in after]}"
    )
    fv, fm = pooled_features(stem, val), pooled_features(stem, val, flip=True)
    with torch.no_grad():
        lv, lm = router.logits(fv.cuda()[:, :, None, None]), router.logits(fm.cuda()[:, :, None, None])
    share = torch.bincount(lv.argmax(1), minlength=4).float() / len(lv)
    t2 = lambda l: torch.sort(l.topk(2, 1).indices, 1).values
    print(
        f"ROUTER val2017: top-1 shares {[round(float(v), 3) for v in share]}; image vs mirror same top-1 {float((lv.argmax(1) == lm.argmax(1)).float().mean()):.4f}, "
        f"same top-2 set {float((t2(lv) == t2(lm)).all(1).float().mean()):.4f}"
    )
    model.cpu()
    for k, name in ((1, "yolo26m-objv1-wb-top1-fixed.pt"), (2, "yolo26m-objv1-wb-pair6-fixed.pt")):
        m = copy.deepcopy(model)
        st = bank_modules(m)[0].state
        st.top_k = k
        bank_modules(m)[0].router.top_k = k
        m.yaml["weight_bank"] = {
            "layers": [6, 22],
            "experts": 4,
            "top_k": k,
            "hidden": 0,
            "eps": 0.0,
            "hard": True,
            "router_fixed": True,
        }
        out = {kk: v for kk, v in ck.items() if kk not in ("model", "ema", "optimizer", "scaler", "updates")}
        out.update(model=m, ema=None, optimizer=None, updates=None, epoch=-1, best_fitness=None)
        torch.save(out, f"{OUT}/{name}")
        print("saved", name, os.path.getsize(f"{OUT}/{name}"))
        w = torch.stack(tuple(bank_modules(m)[5].experts))
        print("  experts identical in bank 5:", bool((w[:, None] - w[None]).abs().max() == 0))
