import sys
import os
import logging
import argparse
import random
import json
import yaml
import easydict
import numpy as np
import torch

import comm
from utils import *
import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data
from itertools import islice
from dataset import RuleDataset, Iterator
import os

class Train_Evaluator(object):

    def __init__(self, model, train_set, valid_set, test_set, optimizer, scheduler=None, gpus=None, num_worker=0):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.num_worker = num_worker

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                logging.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        if self.rank == 0:
            logging.info("Preprocess training set")
        if self.world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if self.device.type == "cuda":
            model = model.cuda(self.device)

        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, batch_per_epoch, smoothing, print_every):
        if comm.get_rank() == 0:
            logging.info('>>>>> Predictor: Training')
        self.train_set.make_batches()
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = torch_data.DataLoader(self.train_set, 1, sampler=sampler, num_workers=self.num_worker)
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device], find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.train()

        total_loss = 0.0
        total_size = 0.0

        sampler.set_epoch(0)

        for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
            all_h, all_r, all_t, target, edges_to_remove = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            target = target.squeeze(0)
            edges_to_remove = edges_to_remove.squeeze(0)
            target_t = torch.nn.functional.one_hot(all_t, self.train_set.graph.entity_size)
            
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                target = target.cuda(device=self.device)
                edges_to_remove = edges_to_remove.cuda(device=self.device)
                target_t = target_t.cuda(device=self.device)
            
            target = target * smoothing + target_t * (1 - smoothing)

            logits, mask = model(all_h, all_r, edges_to_remove)
            if mask.sum().item() != 0:
                logits = (torch.softmax(logits, dim=1) + 1e-8).log()
                loss = -(logits[mask] * target[mask]).sum() / torch.clamp(target[mask].sum(), min=1)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_size += mask.sum().item()
                
            if (batch_id + 1) % print_every == 0:
                if comm.get_rank() == 0:
                    logging.info('{} {} {:.6f} {:.1f}'.format(batch_id + 1, len(dataloader), total_loss / print_every, total_size / print_every))
                total_loss = 0.0
                total_size = 0.0

        if self.scheduler:
            self.scheduler.step()
    
    @torch.no_grad()
    def compute_H(self, print_every):

        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = torch_data.DataLoader(self.train_set, 1, sampler=sampler, num_workers=self.num_worker)
        model = self.model

        model.eval()
        all_H_score = torch.zeros(model.num_rules, device=self.device)
        for batch_id, batch in enumerate(dataloader):
            all_h, all_r, all_t, target, edges_to_remove = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            target = target.squeeze(0)
            edges_to_remove = edges_to_remove.squeeze(0)
            
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                target = target.cuda(device=self.device)
                edges_to_remove = edges_to_remove.cuda(device=self.device)
            
            H, index = model.compute_H(all_h, all_r, all_t, edges_to_remove)
            if H != None and index != None:
                all_H_score[index] += H / len(model.graph.train_facts)
                
            if (batch_id + 1) % print_every == 0:
                if comm.get_rank() == 0:
                    logging.info('{} {}'.format(batch_id + 1, len(dataloader)))
        
        if self.world_size > 1:
            all_H_score = comm.stack(all_H_score)
            all_H_score = all_H_score.sum(0)
        
        return all_H_score.data.cpu().numpy().tolist()
    
    @torch.no_grad()
    def evaluate(self, split, expectation=True):
        if comm.get_rank() == 0:
            logging.info('>>>>> Predictor: Evaluating on {}'.format(split))
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        dataloader = torch_data.DataLoader(test_set, 1, sampler=sampler, num_workers=self.num_worker)
        model = self.model

        model.eval()
        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []
        concat_mask = []
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag = flag.squeeze(0)
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                flag = flag.cuda(device=self.device)

            logits, mask = model(all_h, all_r, None)

            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)
            concat_mask.append(mask)
        
        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)
        concat_mask = torch.cat(concat_mask, dim=0)
        
        ranks = []
        for k in range(concat_all_t.size(0)):
            h = concat_all_h[k]
            r = concat_all_r[k]
            t = concat_all_t[k]
            if concat_mask[k, t].item() == True:
                val = concat_logits[k, t]
                L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
                H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            else:
                L = 1
                H = test_set.graph.entity_size + 1
            ranks += [[h, r, t, L, H]]
        ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)
            
        if self.world_size > 1:
            ranks = comm.cat(ranks)
        
        query2LH = dict()
        for h, r, t, L, H in ranks.data.cpu().numpy().tolist():
            query2LH[(h, r, t)] = (L, H)
            
        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for (L, H) in query2LH.values():
            if expectation:
                for rank in range(L, H):
                    if rank <= 1:
                        hit1 += 1.0 / (H - L)
                    if rank <= 3:
                        hit3 += 1.0 / (H - L)
                    if rank <= 10:
                        hit10 += 1.0 / (H - L)
                    mr += rank / (H - L)
                    mrr += 1.0 / rank / (H - L)
            else:
                rank = H - 1
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 10:
                    hit10 += 1
                mr += rank
                mrr += 1.0 / rank
            
        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)

        return mrr

    def load(self, checkpoint, load_optimizer=True):
        """
        Load a checkpoint from file.
        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
        """

        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(state["model"])

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        comm.synchronize()

    def save(self, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """

        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            torch.save(state, checkpoint)

        comm.synchronize()
