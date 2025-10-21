import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pprint import pprint

import torch
from torch.utils.data import DataLoader
from utils.functions import load_cfg

from neuralsfc.data import AnyDataset
from neuralsfc.model import NeuralSFC


def run(cfg):
    print('Using config:')
    pprint(cfg)

    model = NeuralSFC(cfg)
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

    val_dataset = AnyDataset(
            root=cfg.data_dir,
            dataset=cfg.dataset, 
            train=False,  
            ac_offset=cfg.ac_offset,
            norm_type=cfg.weight_norm,
            out_nc=cfg.n_channels,  #  should be 3, check .yaml file
            e_class=cfg.e_class,
            info=info,
            normalize_e=cfg.normalize_e,
            )

    dataloaders = {
            'train': DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=True, drop_last=True), 
            'val': DataLoader(val_dataset, batch_size=cfg.val_batch_size, num_workers=cfg.n_workers, shuffle=False, drop_last=False) # I change True to False here
        }
    
    # For training
    model.train(dataloaders)

    ## For testing, not available currently
    #model.eval()


if __name__ == "__main__":
    run(load_cfg(base_path='configs/base_config.yml'))
