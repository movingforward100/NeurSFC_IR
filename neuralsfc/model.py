import ast
import logging
import os
import pdb
import re
import shutil
import sys
import time
import warnings
from multiprocessing import Pool
from pprint import pformat
from typing import Dict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import utils.cfg
from basic.butils import img_to_arr, pad_mnist
from basic.context import (apply_mst, build_circuits_graph, build_dual_graph,
                           iter_weights)
from basic.evaluation import (calc_total_variation,
                              fixed_offset_auto_correlation)
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from utils.functions import (copy_params, get_module, get_normalize_func,
                             get_yaml_content, load_params,
                             search_partial_path)
from utils.lzw import get_lzw_length
from utils.misc import (LoHiNormalizer, TwoOneSequential, WeightsAssigner,
                        compute_weights_ac_tv)

from neuralsfc.networks import NACEvaluator, ScalarEvaluator, WeightGenerator
from basic.visualize import draw_sfc
import matplotlib.pyplot as plt

import time
import math


def build_warmup_cosine(optimizer, total_steps, warmup_steps, eta_min=0.0):
    def lr_lambda(step):
        if step < warmup_steps: 
            return float(step + 1) / float(max(1, warmup_steps))

        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return eta_min + (1 - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def notnan(tensor: torch.Tensor):
    assert not torch.any(torch.isnan(tensor)), tensor

def compute_weights_sfc(img_arr, weights):
    if type(img_arr) is torch.Tensor:
        img_arr = img_arr.detach().cpu().numpy()
    else:
        assert type(img_arr) is np.ndarray

    if type(weights) is torch.Tensor:
        weights = weights.detach().cpu().numpy()
    else:
        assert type(weights) is np.ndarray    
    wa = WeightsAssigner(mode='predefined', example=img_arr, weights=weights)
    circuits = build_circuits_graph(img_arr)
    dual_g = build_dual_graph(img_arr, weight_func=wa.predefined_weight)

    #start_time = time.time()
    mst = nx.minimum_spanning_tree(dual_g, algorithm='prim')
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"MST Time: {elapsed_time:.4f} seconds")

    sfc_graph = apply_mst(circuits, mst)
    sfc_graph.remove_edge((0, 0), (0, 1))

    #start_time = time.time()
    sfc = list(nx.shortest_path(sfc_graph, (0, 0), (0, 1)))
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"Compute SFC Time: {elapsed_time:.4f} seconds")
    
    return sfc

def sfc_to_ac_worker(x):
    #start_time = time.time()
    img_arr, sfc, ac_offset = x
    ac = fixed_offset_auto_correlation(img_arr, sfc, ac_offset, True)

    tv = calc_total_variation(img_arr, sfc)
    tv = tv / len(sfc)
    
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"SFC to AC Time: {elapsed_time:.4f} seconds")

    return ac, tv


def weights_to_lzwl_worker(params):
    img_arr, weights = params
    sfc = compute_weights_sfc(img_arr, weights)
    lzw_length = get_lzw_length((img_arr, sfc))
    return lzw_length


def actv_worker(x):
    i, w, ac_offset = x
    return compute_weights_ac_tv(i, w, ac_offset=ac_offset)


def clac_residual_weights(x):
    w, img255 = x
    wa_dafner = WeightsAssigner(mode='mix', example=img255, norm_type='in') # TODO to be fixed here

    # circuits = build_circuits_graph(img255)

    iter_weights(img255, weight_func=wa_dafner.calc_weight)
    wa_dafner.normalize_weights()

    weights_dafner = wa_dafner.weights

    weights = w + weights_dafner
    return weights


class InsNorm1d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(1)

    def forward(self, weights):
        bs, n = weights.shape
        weights = weights.view(bs, 1, n)
        return self.norm(weights).squeeze()
    

class NeuralSFC:
    # The NeuralSFC model.
    @classmethod
    def load_from_record(cls, path):
        yaml_path = os.path.join(path, 'training.log')
        bset_path = os.path.join(path, 'model', 'best_default.pt')
        with open(yaml_path, 'r') as f:
            yaml_lines = f.readlines()
        cfg = ast.literal_eval(get_yaml_content(yaml_lines))
        cfg = EasyDict(cfg)
        cfg.suffix = 'default'
        utils.cfg.cfg_global = cfg
        obj = cls(cfg)
        obj.load_generator_from_ckpt(bset_path)
        return obj


    def __init__(self, cfg) -> None:
        self.cfg = cfg
        device = self.cfg.device
        if cfg.ema_decay != 0:
            self.use_ema = True
            self.ema_decay = cfg.ema_decay
        else:
            self.use_ema = False

        self.normalize_e = cfg.normalize_e

        generator = WeightGenerator(
            n_encoder_layers=cfg.n_encoder_layers_wg, # default: [2, 2, 2, 2] 
            n_img_channels=cfg.n_channels,
            n_embeddings=cfg.n_embeddings,
            large_wg = cfg.large_wg
        )

        if cfg.e_class == 'nac':
            evaluator = NACEvaluator(
                n_encoder_layers=cfg.n_encoder_layers_we, # default: [2, 2, 2, 2]
                n_img_channels=cfg.n_channels,
                n_embeddings=cfg.n_embeddings_evaluator,
            )
        elif cfg.e_class == 'lzwl':
            evaluator = ScalarEvaluator(
                n_encoder_layers=cfg.n_encoder_layers_we, # default: [2, 2, 2, 2]
                n_img_channels=cfg.n_channels,
                n_embeddings=cfg.n_embeddings_evaluator,
                n_regressor_layers=cfg.n_regressor_layers,
            ) 
        else:
            raise NotImplementedError(f'Unknown evaluator class: {cfg.e_class}')

        if self.normalize_e:
            evaluator.scalar_regressor.linear.bias.data = torch.tensor([0.0])
            self.normalize_func = get_normalize_func()
            e_head = nn.Tanh()
            evaluator = TwoOneSequential(evaluator, e_head)
        elif cfg.e_class == 'lzwl':
            
            init_val = 650.0 # 'ucmnist':160.0；ucfmnist':270.0；'ffhq32':650.0  ## we always set this variable to 650 when cfg.e_class == 'lzwl'
            evaluator.scalar_regressor.linear.bias.data = torch.tensor([init_val])
        
        if torch.cuda.device_count() >= 2:
            evaluator = torch.nn.DataParallel(evaluator) 
            generator = torch.nn.DataParallel(generator)     

        self.e = evaluator.to(device)
        self.g = generator.to(device)

        if cfg.weight_norm == 'minmax':
            self.normlize = lambda weights: (weights - weights.detach().min(1, keepdim=True).values) / (weights.detach().max(1, keepdim=True).values - weights.detach().min(1, keepdim=True).values + 1e-5)
        elif cfg.weight_norm == 'in':
            self.normlize = InsNorm1d().to(device)
        else:
            raise NotImplementedError

        self.residual = cfg.residual   # always False 
        if self.residual:
            print('Using residual mode.')

        self.worker_pool = Pool(max(cfg.n_workers, 1))

        self.ac_offset_list = [cfg.ac_offset] * cfg.batch_size   #[ac_offset, ac_offset, ...] # length: B
        

        if isinstance(cfg.ac_offset, list): # multiple k (ac_offset) are used
            # transpose it
            self.ac_offset_list = list(zip(*self.ac_offset_list))
            self.multiple_ac_offset = True
        else: # single k (ac_offset) is used  # we always use single ac_offset here
            self.multiple_ac_offset = False
        
        self.ac_x_offset_list_dict = {}
        for offset in range(1, 10):
            self.ac_x_offset_list_dict[offset] = [offset] * len(self.ac_offset_list)

            # self.ac_x_offset_list_dict = {
            #     1: [1,1,1,1,1,1,1,1],
            #     2: [2,2,2,2,2,2,2,2],
            #     ...
            #     9: [9,9,9,9,9,9,9,9],
            # }
        self.scale_func = None
        self.scale_ac = cfg.scale_ac
        if cfg.scale_ac:   # default: False
            self.scale_func = lambda x: nn.functional.relu((x - 0.75) / (1.0 - 0.75))


    def get_checkpoint(self, info=None):
        checkpoint = {
            'evaluator': get_module(self.e).state_dict(),
            'generator': get_module(self.g).state_dict(),
            'optimizer_e': self.optim_e.state_dict(),
            'optimizer_g': self.optim_g.state_dict(),
            'scheduler_e': self.scheduler_e.state_dict(),
            'scheduler_g': self.scheduler_g.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
        } 
        if info is not None:
            checkpoint['info'] = info

        return checkpoint
    
    def get_checkpoint_no_optimizer(self, info=None):  ## used for Yan Min, reducing weights memory
        checkpoint = {
            'evaluator': get_module(self.e).state_dict(),
            'generator': get_module(self.g).state_dict(),
            'rng_state': torch.get_rng_state(),   
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
        } 
        if info is not None:
            checkpoint['info'] = info
        return checkpoint
    
    def train(self, loaders: DataLoader):

        cfg = self.cfg        
        device = cfg.device

        assert cfg.scheduler_mode in ['step', 'val'], f'unknown lr scheduler mode {cfg.scheduler_mode}'
        self.scheduler_mode = cfg.scheduler_mode  # default: val

        if cfg.e_loss_type == 'mse':  # default setting
            criterion_e = nn.MSELoss(reduction='none').to(device)
        elif cfg.e_loss_type == 'l1':
            criterion_e = nn.L1Loss(reduction='none').to(device)
        else:
            raise NotImplementedError(f'Unknown evaluator loss type: {cfg.e_loss_type}')

        if cfg.g_loss_type == 'mse':
            criterion_g = nn.MSELoss(reduction='none').to(device)
        elif cfg.g_loss_type == 'l1':
            criterion_g = nn.L1Loss(reduction='none').to(device)
        elif cfg.g_loss_type == 'direct':
            criterion_g = lambda y_hat, y: y_hat
        else:
            raise NotImplementedError(f'Unknown evaluator loss type: {cfg.g_loss_type}')

        cfg.model_dir = os.path.join(cfg.output_dir, 'model')
        os.makedirs(cfg.model_dir, exist_ok=True)

        log_path = os.path.join(cfg.output_dir, 'training.log')
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', 
                            filename=log_path, level=logging.INFO)                                                          
        logging.info('\n' + pformat(cfg))
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        self.optim_e = optim.Adam(self.e.parameters(), betas=(cfg.beta1, cfg.beta2), lr=cfg.lr_e, weight_decay=cfg.wd_e)
        self.optim_g = optim.Adam(self.g.parameters(), betas=(cfg.beta1, cfg.beta2), lr=cfg.lr_g, weight_decay=cfg.wd_g)


        if cfg.scheduler == 'exp': # this option is sensitive to the number of epochs
            self.scheduler_e = optim.lr_scheduler.ExponentialLR(self.optim_e, gamma=cfg.lr_decay)
            self.scheduler_g = optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=cfg.lr_decay)
        elif cfg.scheduler == 'cosine':
            from pl_bolts.optimizers.lr_scheduler import \
                LinearWarmupCosineAnnealingLR
            assert self.scheduler_mode == 'step'
            total_steps = len(loaders['train']) * cfg.n_epochs
            warmup_steps = len(loaders['train']) * cfg.warmup * cfg.n_epochs
            self.scheduler_e = LinearWarmupCosineAnnealingLR(self.optim_e, warmup_epochs=warmup_steps, max_epochs=total_steps)
            self.scheduler_g = LinearWarmupCosineAnnealingLR(self.optim_g, warmup_epochs=warmup_steps, max_epochs=total_steps)
        else:
            raise NotImplementedError(f'Unknown scheduler: {cfg.scheduler}')
        

        writer = SummaryWriter(os.path.join(cfg.output_dir, 'tb_log'))

        start_epoch = 1 # TODO to be modified for continue training

        self.best_metric = -10000.0 if cfg.e_class in ['nac'] else 10000.0 # larger is better or smaller is better

        # parameters loading
        if cfg.load_path:  ## default: empty
            path = search_partial_path(cfg.load_path)
            start_epoch = int(os.path.basename(path).split('_')[2]) + 1
            checkpoint = torch.load(path)
            get_module(self.e).load_state_dict(checkpoint['evaluator'])
            get_module(self.g).load_state_dict(checkpoint['generator'])
            self.optim_e.load_state_dict(checkpoint['optimizer_e'])
            self.optim_g.load_state_dict(checkpoint['optimizer_g'])
            self.scheduler_e.load_state_dict(checkpoint['scheduler_e'])
            self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
            torch.set_rng_state(checkpoint['rng_state'])
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
            if self.use_ema:
                # self.avg_param_g = params['generator_avg']
                raise NotImplementedError

        latest_path = os.path.join(cfg.model_dir, f'latest_{cfg.suffix}.pt')
        if cfg.suffix and os.path.exists(latest_path): # try auto-resuming  ## not triggered
            checkpoint = torch.load(latest_path)
            get_module(self.e).load_state_dict(checkpoint['evaluator'])
            get_module(self.g).load_state_dict(checkpoint['generator'])
            self.optim_e.load_state_dict(checkpoint['optimizer_e'])
            self.optim_g.load_state_dict(checkpoint['optimizer_g'])
            self.scheduler_e.load_state_dict(checkpoint['scheduler_e'])
            self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
            torch.set_rng_state(checkpoint['rng_state'])
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
            e_pattern = 'epoch_([0-9]*)_'
            assert 'info' in checkpoint
            start_epoch = int(re.search(e_pattern, checkpoint['info']).group(1)) + 1
            m_pattern = 'best_metric_(-*[0-9]*[.]?[0-9]*)'
            self.best_metric = float(re.search(m_pattern, checkpoint['info']).group(1))

            logging.info(f'Automically resumed from {latest_path}, starting from epoch {start_epoch}.')
            logging.info(f'Previous best metric: {self.best_metric}.')

        self.e.train()
        self.g.train()
        if self.use_ema:
            self.avg_param_g = copy_params(get_module(self.g))

        logging.info('start training...')
        self.last_best_path = None
        self.last_latest_path = None
        # Starting training...
        for epoch in range(start_epoch, cfg.n_epochs + 1):
            self.train_epoch(
                loaders,
                criterion_e,
                criterion_g,
                writer,
                epoch
            )

        writer.close()

    #@track_time
    def train_epoch(
        self,
        loaders: Dict[str, DataLoader],
        criterion_e,
        criterion_g, 
        writer: SummaryWriter,
        epoch, 
        ):

        cfg = self.cfg
        device = cfg.device
        assert not self.scale_ac
        assert not self.use_ema

        info = f'Start train epoch {epoch}, '
        info += f'''lr_e={self.optim_e.param_groups[0]['lr']:.6f}, '''
        info += f'''lr_g={self.optim_g.param_groups[0]['lr']:.6f}, '''
        info += f' for run {cfg.cfg_short_name}'
        logging.info(info)
        step = (epoch - 1) * len(loaders['train'])
        start_time = time.time()
        n_steps_time = None

        # n_pixels = [1024, 4096]
        # train phase
        # TODO to implement this dataloader
        ds = loaders['train'].dataset
        if isinstance(ds, torch.utils.data.dataset.Subset):
            ds = ds.dataset
        if hasattr(ds, 'reset_remapped_index'):
            ds.reset_remapped_index()

        self.e.train()
        self.g.train()

        for batch in tqdm(loaders['train']): 
            #print('10')
            if cfg.val_step_interval > 0:
                self.e.train()
                self.g.train()

            if cfg.e_class in ['nac']:
                img, img255, weights_tgt, weights_dafner, nac_tgt_gt, stv_tgt_gt = batch  ## understand each component in the batch
                nac_tgt_gt, stv_tgt_gt = nac_tgt_gt.to(device), stv_tgt_gt.to(device)
            elif cfg.e_class == 'lzwl':
                img, img255, weights_tgt, weights_dafner, lzwl = batch  ## understand each component in the batch
                lzwl = lzwl.to(device)
            else:
                raise NotImplementedError

            img, weights_tgt = img.to(device), weights_tgt.to(device)  #difference between weights_tgt and weights_dafner
            weights_dafner = weights_dafner.to(device)

            if cfg.e_class == 'nac': 
                y_tgt = nac_tgt_gt
            elif cfg.e_class == 'lzwl':
                y_tgt = lzwl
            else:
                raise NotImplementedError(f'Unknown evaluator class: {cfg.e_class}')
            
            # training evaluator
            self.e.requires_grad_(True)

            weights_gen = self.g(img)    # 输入图片到generaor --> 得到weight, 参考公式1中 weight的定义, 此时generator不更新
            if self.residual:
                weights_gen = weights_gen + weights_dafner
            weights_gen = self.normlize(weights_gen)
            weights_gen_detach = weights_gen.detach()

            if cfg.e_class in ['nac']:
                nac_gen_gt, stv_gen_gt = self.get_nac_stv(img255, weights_gen_detach) ## nac_gen_gt, the ground truth in the training of the evaluator
                nac_gen_avg_gt, stv_gen_avg_gt = self.get_nac_stv_avg(img255, weights_gen_detach)  # using averaged weights to cal nac, stv

                y_gen = nac_gen_gt                  # these are gts for evaluating the evaluator!!!
                y_gen_avg = nac_gen_avg_gt
                
            elif cfg.e_class == 'lzwl':
                if not cfg.only_avg_loss:  ## default: triggered
                    y_gen = self.get_lzw_length(img255, weights_gen_detach)
                y_gen_avg = self.get_lzw_length_avg(img255, weights_gen_detach)
            else:
                raise NotImplementedError

            assert cfg.e_iters == 1 # TODO

            self.optim_e.zero_grad()

            y_tgt_hat = self.e(img, weights_tgt) # it's the prediction of evaluator using weights_tgt， 和generator无关， 直接由batch传入
            y_gen_avg_hat = self.e(img, self.make_avg_batch(weights_gen_detach)) # # it's the prediction of evaluator using average weight
            # y_tgt is also from batch
            
            if cfg.only_avg_loss: 
                e_loss = criterion_e(y_tgt_hat, y_tgt).mean() + criterion_e(y_gen_avg_hat, y_gen_avg).mean()
                e_loss = e_loss / 2.   
            else:
                y_gen_hat = self.e(img, weights_gen_detach) # it's the prediction of evaluator using weights_gen_detach (和generator有关)
                e_loss = criterion_e(y_tgt_hat, y_tgt) + criterion_e(y_gen_hat, y_gen) + criterion_e(y_gen_avg_hat, y_gen_avg)
                e_loss = e_loss.mean()
                e_loss = e_loss * 2 / 3

            e_loss.backward()
            nn.utils.clip_grad_norm_(self.e.parameters(), cfg.max_grad_norm_e)
            self.optim_e.step()

            # training g
            self.e.requires_grad_(False)
            self.optim_g.zero_grad()

            y_hat = self.e(img, self.make_avg_batch(weights_gen))

            g_loss = criterion_g(y_hat, -torch.ones_like(y_tgt))    # g_loss = y_hat
            
            g_loss = g_loss.mean()
            
            g_loss.backward()
            nn.utils.clip_grad_norm_(self.g.parameters(), cfg.max_grad_norm_g)
            self.optim_g.step()

            step += 1

            if step % cfg.print_interval == 0 or self.cfg.debug:
                info = f'epoch: {epoch} train, step: {step}, '
                info += f'e_loss_{cfg.e_class}: {e_loss.item():.5f}, '
                info += f'g_loss: {g_loss.item():.5f}, '
                info += '\n'

                n_pixels =  img.size()[-1] *img.size()[-1] 
                if cfg.e_class in ['nac']:
                    with torch.no_grad():
                        nac_dafner_avg, stv_dafner_avg = self.get_nac_stv_avg(img255, weights_dafner)
                        ac_dafner_avg, tv_dafner_avg = -nac_dafner_avg, stv_dafner_avg * n_pixels  #获取 dafnerg方法的ac, 和 tv
                    
                    
                    
                    ac_apox = -y_hat   ## apox means that the ac value is predicted by the evaluator
                    apox_ac_gain = (ac_apox - ac_dafner_avg).mean().item()
                    info += f'apox_ac_gain: {apox_ac_gain:.5f}, '
                    writer.add_scalar('AC_gain/apox_ac_gain', apox_ac_gain, step)

                    ac_prim = -nac_gen_avg_gt   # ac_prim means that the ac value is calculated using the weight generator
                    real_ac_gain = (ac_prim - ac_dafner_avg).mean().item() #  

                    # tv_prim = stv_gen_gt * n_pixels
                    tv_prim = stv_gen_avg_gt * n_pixels
                    real_tv_gain = (tv_prim - tv_dafner_avg).mean().item()

                    info += f'real_ac_gain: {real_ac_gain:.5f}, '
                    info += f'real_tv_gain: {real_tv_gain:.5f}. '
                    writer.add_scalar('AC_gain/real_ac_gain', real_ac_gain, step)
                    writer.add_scalar('TV_gain/real_tv_gain', real_tv_gain, step)

                elif cfg.e_class == 'lzwl':
                    pass # Do Nothing!

                else:
                    raise NotImplementedError(f'Unknown evaluator class: {cfg.e_class}')

                logging.info(info)
                writer.add_scalar('Loss/generator', g_loss.item(), step)
                writer.add_scalar('Loss/evaluator', e_loss.item(), step)
                writer.add_scalar('LR/lr_g', self.scheduler_g.get_last_lr()[0], step)
                writer.add_scalar('LR/lr_e', self.scheduler_e.get_last_lr()[0], step)
            if self.scheduler_mode == 'step':
                self.scheduler_e.step()
                self.scheduler_g.step()
    
        logging.info(f'epoch: {epoch} train finished.')
        # val phase
        logging.info('start validating...')
        self.e.eval()
        self.g.eval()
        # backup parameters
        if self.use_ema:
            backup_params = copy_params(get_module(self.g))
            load_params(get_module(self.g), self.avg_param_g)

        running_g_loss = 0.
        running_apox_ac_gain = 0.
        running_apox_tv_gain = 0.
        running_real_ac_gain = 0.
        running_real_tv_gain = 0.
        running_real_ac = 0.
        N_val = 0

        if self.cfg.show_lzw:  # this will be true if the e_class is lzwl
            running_lzw_len = 0.

        if self.cfg.show_more_acs:
            running_ac_x_dict = {}
            for offset in range(1,4):
                running_ac_x_dict[offset] = 0.0

        # for img, img255, weights_tgt, weights_dafner, nac_tgt_gt, stv_tgt_gt, ac_dafner, tv_dafner in tqdm(loaders['val']):
        for batch in tqdm(loaders['val']):
                
            if cfg.e_class in ['nac']:
                img, img255, weights_tgt, weights_dafner, nac_tgt_gt, stv_tgt_gt = batch
                nac_tgt_gt, stv_tgt_gt = nac_tgt_gt.to(device), stv_tgt_gt.to(device)
            elif cfg.e_class == 'lzwl':
                img, img255, weights_tgt, weights_dafner, lzwl = batch
                lzwl = lzwl.to(device)
            else:
                raise NotImplementedError

            img = img.to(device)
            n_pixels_val =  img.shape[2] * img.shape[3]
            weights_dafner = weights_dafner.to(device)

            bs = img.size(0)
            N_val += bs

            with torch.no_grad():
                weights_gen = self.g(img)
                if self.residual:
                    weights_gen = weights_gen + weights_dafner
                weights_gen = self.normlize(weights_gen)

                nac_gen_avg_gt, stv_gen_avg_gt = self.get_nac_stv_avg(img255, weights_gen)
                nac_dafner_avg, stv_dafner_avg = self.get_nac_stv_avg(img255, weights_dafner)

                if self.cfg.show_lzw:
                    lzw_len = self.get_lzw_length_avg(img255, weights_gen, force_not_normalize=True).sum().item()
                    running_lzw_len += lzw_len

                if self.cfg.show_more_acs:
                    for offset in range(1,4):
                        running_ac_x_dict[offset] +=  (-self.get_nac_stv_avg(img255, weights_gen, offset=offset)[0]).sum().item()

                ac_dafner_avg, tv_dafner_avg = -nac_dafner_avg, stv_dafner_avg * n_pixels_val

                if self.scale_ac:
                    nac_gen_avg_gt = -self.scale_func(-nac_gen_avg_gt)
                y_hat = self.e(img, weights_gen)
        
                if cfg.e_class == 'nac':
                    ac_apox = -y_hat
                    apox_ac_gain = (ac_apox - ac_dafner_avg).sum().item()
                    running_apox_ac_gain += apox_ac_gain

                ac_prim = -nac_gen_avg_gt
                real_ac_gain = (ac_prim - ac_dafner_avg).sum().item()
                running_real_ac_gain += real_ac_gain

                tv_prim = stv_gen_avg_gt * n_pixels_val
                real_tv_gain = (tv_prim - tv_dafner_avg).sum().item()
                running_real_tv_gain += real_tv_gain

                real_ac = ac_prim.sum().item()
                running_real_ac += real_ac

            if self.cfg.debug:
                break

        
        val_g_loss = running_g_loss / N_val 
        val_apox_ac_gain = running_apox_ac_gain / N_val
        val_apox_tv_gain = running_apox_tv_gain / N_val
        val_real_ac_gain = running_real_ac_gain / N_val 
        val_real_tv_gain = running_real_tv_gain / N_val
        val_real_ac = running_real_ac / N_val 

        if self.cfg.show_lzw:
            val_lzw_len = running_lzw_len / N_val

        if self.cfg.show_more_acs:
            val_ac_x_dict = {}
            for offset in range(1,4):
                val_ac_x_dict[offset] = running_ac_x_dict[offset] / N_val

        logging.info(f'epoch: {epoch} val finished. ')

        info = f'val_g_loss: {val_g_loss:.5f}, '
        info += f'val_apox_ac_gain: {val_apox_ac_gain:.5f}, '
        info += f'val_apox_tv_gain: {val_apox_tv_gain:.5f}, '
        info += f'val_real_ac_gain: {val_real_ac_gain:.5f}, '
        info += f'val_real_tv_gain: {val_real_tv_gain:.5f}. '
        info += f'val_real_ac: {val_real_ac:.5f}. '

        if self.cfg.show_lzw:
            info += f'val_lzw_len: {val_lzw_len}. '

        if self.cfg.show_more_acs:
            for offset in range(1,4):
                info += f'val_ac_{offset}: {val_ac_x_dict[offset]}. '

        logging.info(info)

        writer.add_scalar('Val/val_g_loss', val_g_loss, step)
        writer.add_scalar('Val/val_apox_ac_gain', val_apox_ac_gain, step)
        writer.add_scalar('Val/val_apox_tv_gain', val_apox_tv_gain, step)
        writer.add_scalar('Val/val_real_ac_gain', val_real_ac_gain, step)
        writer.add_scalar('Val/val_real_tv_gain', val_real_tv_gain, step)
        writer.add_scalar('Val/val_real_ac', val_real_ac, step)

        if self.cfg.show_lzw:
            writer.add_scalar('Val/val_lzw_len', val_lzw_len, step)

        if self.cfg.show_more_acs:
            for offset in range(1,4):
                writer.add_scalar(f'Val/val_ac_{offset}', val_ac_x_dict[offset], step)

        epoch_duration = time.time() - start_time
        logging.info("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

        
        print('Saving model and state...')


        if cfg.e_class in ['nac']:
            best_cond = val_real_ac_gain > self.best_metric
            cur_metric = val_real_ac_gain
        else:
            assert self.cfg.show_lzw and cfg.e_class == 'lzwl'
            best_cond = val_lzw_len < self.best_metric
            cur_metric= val_lzw_len


        info_str = f'epoch_{epoch}_step_{step}_gloss_{val_g_loss:.3f}_racgain_{val_real_ac_gain:.3f}_rtvgain_{val_real_tv_gain:.3f}_best_metric_{cur_metric}'
        if cfg.suffix: # auto-resuming mode enabled. save name with suffix.
            latest_path = os.path.join(cfg.model_dir, f'latest_{cfg.suffix}.pt')
        else: # auto-resuming mode disabled. save verbose name.
            latest_path = os.path.join(cfg.model_dir, f'latest_{info_str}.pt')

        if self.last_best_path is not None:
            os.remove(self.last_latest_path)
        torch.save(self.get_checkpoint(info=info_str), latest_path)
        self.last_latest_path = latest_path


        if best_cond:
            self.best_metric = cur_metric
            if cfg.suffix: # auto-resuming mode enabled. save name with suffix.
                best_path = os.path.join(cfg.model_dir, f'best_{cfg.suffix}.pt')
            else: # auto-resuming mode disabled. save verbose name.
                best_path = os.path.join(cfg.model_dir, f'best_{info_str}.pt')

            if self.last_best_path is not None:
                os.remove(self.last_best_path)
            shutil.copy2(latest_path, best_path)
            self.last_best_path = best_path


        if not cfg.only_save_best:
            if (cfg.checkpoint_epochs != 0 and epoch % cfg.checkpoint_epochs == 0) or epoch == cfg.n_epochs - 1:
                print('Saving model and state...')
                if self.use_ema:
                    raise NotImplementedError
                info_str = f'epoch_{epoch}_step_{step}_gloss_{val_g_loss:.3f}_racgain_{val_real_ac_gain:.3f}_rtvgain_{val_real_tv_gain:.3f}'
                torch.save(
                    self.get_checkpoint(info=info_str),
                    os.path.join(cfg.model_dir, f'{info_str}.pt')
                )


        writer.add_scalar('Epoch', epoch, step)
        # lr_scheduler should be called at end of epoch
        if self.scheduler_mode == 'val':
            self.scheduler_e.step()
            self.scheduler_g.step()

    #@track_time
    def get_nac_stv(self, img255, weights):
        assert type(img255) is torch.Tensor
        img255 = img255.numpy()

        weights = weights.detach().cpu().numpy()
        assert not np.any(np.isnan(weights)), weights

        if self.multiple_ac_offset: # multiple k (ac_offset) are used
            acstv_list = [self.worker_pool.map(actv_worker, zip(img255, weights, single_list)) for single_list in self.ac_offset_list]
            acstv = torch.mean(torch.tensor(acstv_list).type(torch.FloatTensor).to(self.cfg.device), dim=0).t()
        else: # single k (ac_offset) is used
            acstv = self.worker_pool.map(actv_worker, zip(img255, weights, self.ac_offset_list))
            acstv = torch.tensor(acstv).type(torch.FloatTensor).to(self.cfg.device).t()
        ac, stv = acstv
        nac = -ac
        return nac, stv

    #@track_time
    def get_nac_stv_avg(self, img255, weights, offset=None):
        
        if offset is None:
            ac_offset_list = self.ac_offset_list
        else:
            assert isinstance(offset, int) and offset in range(1, 10)
            ac_offset_list = self.ac_x_offset_list_dict[offset]

        assert type(img255) is torch.Tensor
        img255 = img255.numpy()

        
        weights = weights.detach().cpu().numpy()
        weights = np.asarray(weights)

        if weights.ndim == 1:
            weights = weights[None, :]        # (1, N)

        weights_avg = weights.mean(axis=0)    ### key operation: average the weights for batch
        assert not np.any(np.isnan(weights)), weights

        avg_sfc = compute_weights_sfc(img255[0], weights_avg) # ac_offset_list[0]
        avg_sfc_list = [avg_sfc] * self.cfg.batch_size

        if self.multiple_ac_offset and offset is None: # multiple k (ac_offset) are used
            acstv_list = [self.worker_pool.map(sfc_to_ac_worker, zip(img255, avg_sfc_list, single_list)) for single_list in ac_offset_list]
            acstv = torch.mean(torch.tensor(acstv_list).type(torch.FloatTensor).to(self.cfg.device), dim=0).t()
        else: # single k (ac_offset) is used
            acstv = self.worker_pool.map(sfc_to_ac_worker, zip(img255, avg_sfc_list, ac_offset_list))
            acstv = torch.tensor(acstv).type(torch.FloatTensor).to(self.cfg.device).t()

        ac, stv = acstv
        nac = -ac
        return nac, stv


    def get_lzw_length(self, img255, weights):
        assert type(img255) is torch.Tensor
        img255 = img255.numpy()

        weights = weights.detach().cpu().numpy()
        lzw_length_list = self.worker_pool.map(weights_to_lzwl_worker, zip(img255, weights))
        lzw_length = torch.tensor(lzw_length_list).type(torch.FloatTensor)

        if self.normalize_e:
            lzw_length = self.normalize_func(lzw_length)

        return lzw_length.to(self.cfg.device)


    def get_lzw_length_avg(self, img255, weights, force_not_normalize=False):
        assert type(img255) is torch.Tensor
        img255 = img255.numpy()

        weights = weights.detach().cpu().numpy()
        weights = np.asarray(weights)

        if weights.ndim == 1:
            weights = weights[None, :]        # (1, N)

        weights_avg = weights.mean(axis=0)
        assert not np.any(np.isnan(weights)), weights
        avg_sfc = compute_weights_sfc(img255[0], weights_avg) #, self.ac_offset_list[0]
        avg_sfc_list = [avg_sfc] * self.cfg.batch_size

        # lzw_length_avg = np.mean(self.worker_pool.map(get_lzw_length, zip(img255, avg_sfc_list)))
        lzw_length_list = self.worker_pool.map(get_lzw_length, zip(img255, avg_sfc_list))
        lzw_length = torch.tensor(lzw_length_list).type(torch.FloatTensor)
        if self.normalize_e and not force_not_normalize:
            lzw_length = self.normalize_func(lzw_length)
        return lzw_length.to(self.cfg.device)

    
    def make_avg_batch(self, batch: torch.tensor):
        assert type(batch) is torch.Tensor
        sz = batch.size()
        batch = batch.mean(0, keepdim=True).expand(*sz).contiguous()

        return batch

    def load_generator_from_ckpt(self, ckpt_path):
        print(f'Loading checkpoint from {ckpt_path}...')
        checkpoint = torch.load(ckpt_path)
        get_module(self.g).load_state_dict(checkpoint['generator'])

    ## I will add eval function later
    #def eval()
        
        

