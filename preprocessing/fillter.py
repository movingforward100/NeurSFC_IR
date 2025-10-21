# save as tools/filter_patches.py
import os, math, shutil
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

def shannon_entropy_u8(img_u8):
    # img_u8: (H,W,3) uint8
    g = (0.299*img_u8[...,0] + 0.587*img_u8[...,1] + 0.114*img_u8[...,2]).astype(np.uint8)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist / max(1, g.size)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())  # [0, 8] bits

def is_informative(img_u8, mean_thr=3.0, std_thr=2.0, ent_thr=0.5):
    """
    默认规则（满足任一则判为“无信息”，会被剔除）：
      灰度均值 < 3  或  灰度标准差 < 2  或  熵 < 0.5 bits
    你也可以只用 mean/std，把 ent_thr 设为 None。
    """
    g = (0.299*img_u8[...,0] + 0.587*img_u8[...,1] + 0.114*img_u8[...,2])
    m, s = float(g.mean()), float(g.std())
    if m < mean_thr or s < std_thr:
        return False
    if ent_thr is not None:
        ent = shannon_entropy_u8(img_u8)
        if ent < ent_thr:
            return False
    return True

def filter_folder(src_dir, keep_dir=None, rej_dir=None,
                  mean_thr=4.0, std_thr=2.0, ent_thr=0.5,
                  exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
    src_dir = os.path.abspath(src_dir)
    if keep_dir is None: keep_dir = src_dir + "_filtered"
    if rej_dir  is None: rej_dir  = src_dir + "_removed"
    os.makedirs(keep_dir, exist_ok=True)
    os.makedirs(rej_dir,  exist_ok=True)

    files = sorted([p for p in glob(os.path.join(src_dir, "*")) if p.lower().endswith(exts)])
    kept = 0; dropped = 0
    means, stds = [], []

    for p in tqdm(files, desc="Filtering"):
        try:
            img = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
            g = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2])
            means.append(float(g.mean())); stds.append(float(g.std()))
            if is_informative(img, mean_thr, std_thr, ent_thr):
                #shutil.move(p, os.path.join(keep_dir, os.path.basename(p)))
                kept += 1
            else:
                #shutil.move(p, os.path.join(rej_dir, os.path.basename(p)))
                dropped += 1
        except Exception as e:
            print(f"[warn] skip {p}: {e}")

    means = np.array(means); stds = np.array(stds)
    print(f"\nDone. total={len(files)} kept={kept} dropped={dropped}")
    if len(means):
        print(f"mean(gray)  p5={np.percentile(means,5):.2f}  median={np.median(means):.2f}  p95={np.percentile(means,95):.2f}")
        print(f"std(gray)   p5={np.percentile(stds,5):.2f}  median={np.median(stds):.2f}  p95={np.percentile(stds,95):.2f}")
    print(f"kept dir: {keep_dir}\nrejected dir: {rej_dir}")

if __name__ == "__main__":
    # 修改为patch 目录（例如 dataset/LOLv1/train/LQ_crops_64x64）
    SRC = "dataset/GoPro/train/LQ_crops_64x64_overlap_0"
    filter_folder(SRC, mean_thr=4.0, std_thr=2.0, ent_thr=0.5)
