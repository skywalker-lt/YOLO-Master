"""Dense splice H: the YOLO26-M stem (layers 0-5) in front of the YOLO26-L tail (layers 6-23).

M and L share widths; L doubles the depth, so the splice is yolo26.yaml at scale l with the stem's block repeats at the
M depth. Writes the architecture YAML (ultralytics/cfg/models/26/yolo26l-mstem.yaml) and a checkpoint whose weights
are copied name-for-name from the two public checkpoints (strict load, so any architecture mismatch fails loudly).
usage: python splice_build.py <yolo26m.pt> <yolo26l.pt> <out.pt>"""

import sys
from pathlib import Path

import torch
import yaml

from ultralytics.nn.tasks import DetectionModel

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "ultralytics/cfg/models/26"
STEM = 6  # layers 0-5 come from M


def write_yaml():
    base = yaml.safe_load((CFG / "yolo26.yaml").read_text())
    depth_m, depth_l = base["scales"]["m"][0], base["scales"]["l"][0]
    for i in range(STEM):
        layer = base["backbone"][i]
        if isinstance(layer[1], int) and layer[1] > 1:  # repeats, written so that scale l yields the M depth
            layer[1] = max(round(layer[1] * depth_m / depth_l), 1)
    base["scales"] = {"l": base["scales"]["l"]}
    out = CFG / "yolo26l-mstem.yaml"
    out.write_text(
        "# Dense splice H (DS-YOLO Phase 2): YOLO26-M stem (layers 0-5 at M depth) + YOLO26-L tail. Built by\n"
        "# scripts/ds_yolo/splice_build.py from yolo26.yaml; only scale l is defined.\n"
        + yaml.safe_dump(base, sort_keys=False)
    )
    return out


def load(pt):
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    return ck, (ck.get("ema") or ck["model"]).float().eval()


if __name__ == "__main__":
    m_pt, l_pt, out_pt = sys.argv[1:4]
    cfg = write_yaml()
    ckm, m = load(m_pt)
    ckl, lm = load(l_pt)
    splice = DetectionModel(str(cfg), ch=3, nc=lm.yaml["nc"], verbose=False)
    sd = {}
    for k, v in m.state_dict().items():
        if int(k.split(".")[1]) < STEM:
            sd[k] = v.clone()
    for k, v in lm.state_dict().items():
        if int(k.split(".")[1]) >= STEM:
            sd[k] = v.clone()
    splice.load_state_dict(sd, strict=True)
    splice.names, splice.args = lm.names, getattr(lm, "args", None)
    splice.eval()
    with torch.no_grad():  # the stem is M's and the tail is L's, bit for bit
        x = torch.randn(1, 3, 128, 128)
        ms = x
        for layer in list(m.model)[:STEM]:
            ms = layer(ms)
        ss = x
        for layer in list(splice.model)[:STEM]:
            ss = layer(ss)
        assert torch.equal(ms, ss), "stem differs from M"
    params = lambda mm: sum(p.numel() for p in mm.parameters())
    print(f"splice parameters {params(splice):,} (M {params(m):,}, L {params(lm):,}); yaml {cfg}")
    out = {k: v for k, v in ckl.items() if k not in ("model", "ema", "optimizer", "scaler", "updates")}
    out.update(model=splice, ema=None, optimizer=None, updates=None, epoch=-1, best_fitness=None)
    torch.save(out, out_pt)
    print("saved", out_pt)
