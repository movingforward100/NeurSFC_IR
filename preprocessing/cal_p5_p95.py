import os
import numpy as np
from PIL import Image
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.lzw import get_lzw_length

import cv2
import random

from basic.butils import img_to_arr

from utils.functions import load_cfg
from neuralsfc.data import AnyDataset

from natsort import natsorted
from glob import glob


def cal_p5_p95(cfg):

    info = cfg.info if hasattr(cfg, 'info') else None
    train_dataset = AnyDataset(
            root=cfg.data_dir, 
            dataset=cfg.dataset,
            train=True,
            ac_offset=cfg.ac_offset, 
            norm_type=cfg.weight_norm,
            out_nc=cfg.n_channels,  #  should be 3, check .yaml file
            n_select=cfg.n_select,
            e_class=cfg.e_class,
            info=info,
            normalize_e=cfg.normalize_e,
            )
    
    root = cfg.data_dir
    input_paths = natsorted(glob(os.path.join(root, 'train', 'LQ_crops_64x64_skip_128', '*.*')))

    vals = []
    
    for image_path, index in zip(input_paths, range(len(input_paths))):
        print(image_path)

        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        img255 = img_to_arr(img)  #np.float64
        #print(type(img255), img255.dtype, np.min(img255), np.max(img255))
        _, _, lzwl = train_dataset.get_weights_lzwl(img255, info=info)

        vals.append(lzwl)

    print(len(vals))
    vals = np.asarray(vals, dtype=np.float64)
    p5, p95 = np.percentile(vals, [5, 95])
    print(f"count={len(vals)}  min={vals.min():.1f}  p5={p5:.1f}  median={np.median(vals):.1f}  p95={p95:.1f}  max={vals.max():.1f}")


if __name__ == "__main__":
    # command: 
    # python preprocessing/cal_p5_p95.py --cfg configs/IR/Deblurring/Gopro_lzwl.yml
    # python preprocessing/cal_p5_p95.py --cfg configs/IR/LowLight/LOL_lzwl.yml
    cal_p5_p95(load_cfg(base_path='configs/base_config.yml'))
