import os
import random

import networkx as nx
import numpy as np
import torch
import basic.butils as butils
from basic.butils import img_to_arr
from basic.context import (apply_mst, build_circuits_graph, build_dual_graph,
                           iter_weights)
from basic.evaluation import (calc_total_variation,
                              fixed_offset_auto_correlation,
                              multi_offset_auto_correlation)
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from tqdm import tqdm
from utils.functions import get_normalize_func
from utils.lzw import get_lzw_length, get_centroids
from utils.misc import WeightsAssigner
from hashlib import md5
import subprocess
import gdown
import glob
from PIL import Image
import cv2
import time
from natsort import natsorted
from glob import glob

# def track_time(func):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         print(f'{func.__name__} started')
#         result = func(*args, **kwargs)
#         end = time.time()
#         print(f"⏱️ {func.__name__} took {end - start:.4f} seconds.")
#         return result
#     return wrapper

def notnan(tensor: torch.Tensor):
    assert not torch.any(torch.isnan(tensor)), tensor


class BaseDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.root = None
        self.ac_offset = None
        self.multiple_ac_offset = None
        self.train = None
        self.norm_type = None
        self.data = None
        self._dataset_specific_transform = None
        self.length = None

    def calc_dafner_ac_tv(self, img255):
        wa = WeightsAssigner(mode='predefined', example=img255, norm_type=self.norm_type)  ## 

        circuits = build_circuits_graph(img255)
        dual_g = build_dual_graph(img255, weight_func=wa.calc_weight)

        mst = nx.minimum_spanning_tree(dual_g, algorithm='prim')
        sfc_graph = apply_mst(circuits, mst)

        sfc_graph.remove_edge((0, 0), (0, 1))
        sfc = list(nx.all_simple_paths(sfc_graph, (0, 0), (0, 1)))[0]

        # ac = fixed_offset_auto_correlation(img255, sfc, self.ac_offset, True)
        if self.multiple_ac_offset:
            ac = multi_offset_auto_correlation(img255, sfc, self.ac_offset, True)
        else:
            ac = fixed_offset_auto_correlation(img255, sfc, self.ac_offset, True)  
        ac = torch.tensor(ac).type(torch.FloatTensor)

        tv = calc_total_variation(img255, sfc)   ## 计算 tv
        tv = torch.tensor(tv).type(torch.FloatTensor)

        return ac, tv

    #@track_time
    def get_weights_sfc(self, img255, info=None):
        wa = WeightsAssigner(mode='mix', example=img255, norm_type=self.norm_type)  
 
        circuits = build_circuits_graph(img255)

        iter_weights(img255, weight_func=wa.calc_weight)
        wa.normalize_weights()

        weights_dafner = wa.weights.copy() 

        
        dual_g = build_dual_graph(img255, weight_func=wa.predefined_weight)

        mst = nx.minimum_spanning_tree(dual_g, algorithm='prim')
        sfc_graph = apply_mst(circuits, mst)

        sfc_graph.remove_edge((0, 0), (0, 1))
        
        #start_time = time.time()
        sfc = list(nx.shortest_path(sfc_graph, (0, 0), (0, 1)))
        #end_time = time.time()
        #elapsed_time = end_time - start_time
        #print(f"Simple Paths Time: {elapsed_time:.4f} seconds")

        return wa.weights, weights_dafner, sfc

    def calc_nac_stv(self, img255, sfc, info=None):
        # ac = fixed_offset_auto_correlation(img255, sfc, self.ac_offset, True)
        if self.multiple_ac_offset:
            ac = multi_offset_auto_correlation(img255, sfc, self.ac_offset, True, info=info)
        else:
            ac = fixed_offset_auto_correlation(img255, sfc, self.ac_offset, True, info=info)
        nac = -torch.tensor(ac).type(torch.FloatTensor)   ## negative  auto correlation

        notnan(nac)

        tv = calc_total_variation(img255, sfc)
        stv = tv / len(sfc)
        # assert len(sfc) == 1024 or len(sfc) == 4096 # Only for 32 x 32 image or 64 x 64
        stv = torch.tensor(stv).type(torch.FloatTensor)

        return nac, stv

    #@track_time
    def get_weights_nac_stv(self, img255, info=None):
        weights, weights_dafner, sfc = self.get_weights_sfc(img255, info)
        weights = torch.tensor(weights).type(torch.FloatTensor)
        weights_dafner = torch.tensor(weights_dafner).type(torch.FloatTensor)
        nac, stv = self.calc_nac_stv(img255, sfc, info)
        return weights, weights_dafner, nac, stv
    #@track_time
    def get_weights_nac_stv_lzwl(self, img255, info=None):
        weights, weights_dafner, sfc = self.get_weights_sfc(img255, info)
        weights = torch.tensor(weights).type(torch.FloatTensor)
        weights_dafner = torch.tensor(weights_dafner).type(torch.FloatTensor)
        nac, stv = self.calc_nac_stv(img255, sfc, info)
        lzw_len = get_lzw_length((img255, sfc))    ## 
        return weights, weights_dafner, nac, stv, torch.tensor(lzw_len).type(torch.FloatTensor)
    #@track_time
    def get_weights_lzwl(self, img255, info=None):    
        weights, weights_dafner, sfc = self.get_weights_sfc(img255, info)
        weights = torch.tensor(weights).type(torch.FloatTensor)
        weights_dafner = torch.tensor(weights_dafner).type(torch.FloatTensor)
        lzw_len = get_lzw_length((img255, sfc))
        return weights, weights_dafner, torch.tensor(lzw_len).type(torch.FloatTensor)


    def __len__(self) -> int:
        return self.length


class AnyDataset(BaseDataset):
    def __init__(self, root, dataset, ac_offset=3, train=True, norm_type='in', out_nc=None, e_class='nac', **kargs) -> None:
        super().__init__()
        self.root = root
        self.ac_offset = ac_offset
        self.e_class=e_class
        if isinstance(self.ac_offset, list):
            self.multiple_ac_offset = True
        else:
            self.multiple_ac_offset = False
        self.train = train
        self.norm_type = norm_type
        self.dataset = dataset
        self.source_nc = None
        self.out_nc = out_nc
        t_list = []
        self.normalize_e = False
        self.pil_transform = None

        self.kargs = kargs
        if self.e_class == 'lzwl':
            assert 'normalize_e' in self.kargs
            self.normalize_e = kargs['normalize_e']
            assert isinstance(self.normalize_e, bool)
            if self.normalize_e:
                self.normalize_func = get_normalize_func()

        self.source_nc = 3
        self.target_size = kargs.get('target_size', 64)   ### 训练默认设置  64*64
        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self._dataset_specific_transform = transforms.Compose(t_list)
        if train:
            self.root = os.path.join(self.root, 'train')
        else:
            self.root = os.path.join(self.root, 'test')
        print(self.root)
        self.input_paths = natsorted(glob(os.path.join(self.root, 'LQ_crops_64x64_skip_128', '*.*')))
        print('length of input paths: ' + str(len(self.input_paths)))

        self.length = len(self.input_paths)
        

    def __getitem__(self, index: int):
        
        img = cv2.cvtColor(cv2.imread(self.input_paths[index]), cv2.COLOR_BGR2RGB)
        
        img255 = img_to_arr(img)  #np.float64
        img = self._dataset_specific_transform(img) # torch.Tensor

        img_size = img255.shape[-2]

        if self.e_class == 'nac':
            weights, weights_dafner, nac, stv = self.get_weights_nac_stv(img255)
            if self.out_nc is not None and (self.source_nc != self.out_nc):
                if self.source_nc == 1 and self.out_nc == 3:
                    img = img.repeat(3, 1, 1)
                    img255 = img255.reshape(img_size, img_size, 1).repeat(3, axis=-1)

            batch = img, img255, weights, weights_dafner, nac, stv

        elif self.e_class == 'lzwl':
            info = self.kargs['info'] if 'info' in self.kargs else None
            # pdb.set_trace()
            weights, weights_dafner, lzwl = self.get_weights_lzwl(img255, info=info)
            if self.normalize_e:
                lzwl = self.normalize_func(lzwl)
            batch = img, img255, weights, weights_dafner, lzwl
        else:
            raise NotImplementedError
        return batch


def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

def random_crop(lr, size_lr):

    size_lr_x = lr.shape[0]
    size_lr_y = lr.shape[1]

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # LR Patch
    lr_patch = lr[start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr, :]

    return lr_patch

