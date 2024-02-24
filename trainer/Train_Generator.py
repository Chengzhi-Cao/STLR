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

class Train_Generator(object):

    def __init__(self, model, gpu):
        self.model = model

        if gpu is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(gpu)

        model = model.cuda(self.device)
    
    def train(self, rule_set, num_epoch=10000, lr=1e-3, print_every=100, batch_size=512):
        if comm.get_rank() == 0:
            logging.info('>>>>> Generator: Training')
        model = self.model
        model.train()
        
        dataloader = torch_data.DataLoader(rule_set, batch_size, shuffle=True, collate_fn=RuleDataset.collate_fn)
        iterator = Iterator(dataloader)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        total_loss = 0.0
        for epoch in range(num_epoch):
            batch = next(iterator)
            inputs, target, mask, weight = batch
            hidden = self.zero_state(inputs.size(0))
            
            if self.device.type == "cuda":
                inputs = inputs.cuda(self.device)
                target = target.cuda(self.device)
                mask = mask.cuda(self.device)
                weight = weight.cuda(self.device)

            loss = model.loss(inputs, target, mask, weight, hidden)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if (epoch + 1) % print_every == 0:
                if comm.get_rank() == 0:
                    logging.info('{} {} {:.6f}'.format(epoch + 1, num_epoch, total_loss / print_every))
                total_loss = 0.0
    
    def zero_state(self, batch_size): 
        state_shape = (self.model.num_layers, batch_size, self.model.hidden_dim)
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False, device=self.device)
        return (h0, c0)
    
    @torch.no_grad()
    def log_probability(self, rules):
        if rules == []:
            return []
        
        model = self.model
        model.eval()

        rules = [rule + [model.ending_idx] for rule in rules]
        max_len = max([len(rule) for rule in rules])
        for k in range(len(rules)):
            rule_len = len(rules[k])
            for i in range(max_len - rule_len):
                rules[k] += [model.padding_idx]
        rules = torch.tensor(rules, dtype=torch.long, device=self.device)
        inputs = rules[:, :-1]
        target = rules[:, 1:]
        n, l = target.size(0), target.size(1)
        mask = (target != model.padding_idx)
        hidden = self.zero_state(inputs.size(0))
        logits, hidden = model(inputs, inputs[:, 0], hidden)
        logits = torch.log_softmax(logits, -1)
        logits = logits * mask.unsqueeze(-1)
        target = (target * mask).unsqueeze(-1)
        log_prob = torch.gather(logits, -1, target).squeeze(-1) * mask
        log_prob = log_prob.sum(-1)
        return log_prob.data.cpu().numpy().tolist()

    @torch.no_grad()
    def next_relation_log_probability(self, seq, temperature):
        model = self.model
        model.eval()

        inputs = torch.tensor([seq], dtype=torch.long, device=self.device)
        relation = torch.tensor([seq[0]], dtype=torch.long, device=self.device)
        hidden = self.zero_state(1)
        logits, hidden = model(inputs, relation, hidden)
        log_prob = torch.log_softmax(logits[0, -1, :] / temperature, dim=-1).data.cpu().numpy().tolist()
        return log_prob
    
    @torch.no_grad()
    def beam_search(self, num_samples, max_len, temperature=0.2):
        if comm.get_rank() == 0:
            logging.info('>>>>> Generator: Rule generation with beam search')
        model = self.model
        model.eval()
        
        max_len += 1
        all_rules = []
        for relation in range(model.num_relations):
            found_rules = []
            prev_rules = [[[relation], 0]]
            for k in range(max_len):
                current_rules = list()
                for _i, (rule, score) in enumerate(prev_rules):
                    assert rule[-1] != model.ending_idx
                    log_prob = self.next_relation_log_probability(rule, temperature)
                    for i in (range(model.label_size) if (k + 1) != max_len else [model.ending_idx]):
                        new_rule = rule + [i]
                        new_score = score + log_prob[i]
                        (current_rules if i != model.ending_idx else found_rules).append((new_rule, new_score))
                    
                prev_rules = sorted(current_rules, key=lambda x:x[1], reverse=True)[:num_samples]
                found_rules = sorted(found_rules, key=lambda x:x[1], reverse=True)[:num_samples]

            ret = [rule[0:-1] + [score] for rule, score in found_rules]
            all_rules += ret
        return all_rules
    
    @torch.no_grad()
    def sample(self, num_samples, max_len, temperature=1.0):
        if comm.get_rank() == 0:
            logging.info('>>>>> Generator: Rule generation with sampling')
        model = self.model
        model.eval()

        all_rules = []
        for relation in range(model.num_relations):
            rules = torch.zeros([num_samples, max_len + 1], dtype=torch.long, device=self.device) + model.ending_idx
            log_probabilities = torch.zeros([num_samples, max_len + 1], device=self.device)
            head = torch.tensor([relation for k in range(num_samples)], dtype=torch.long, device=self.device)

            rules[:, 0] = relation
            hidden = self.zero_state(num_samples)

            for pst in range(max_len):
                inputs = rules[:, pst].unsqueeze(-1)
                logits, hidden = model(inputs, head, hidden)
                logits /= temperature
                log_probability = torch.log_softmax(logits.squeeze(1), dim=-1)
                probability = torch.softmax(logits.squeeze(1), dim=-1)
                sample = torch.multinomial(probability, 1)
                log_probability = log_probability.gather(1, sample)

                mask = (rules[:, pst] != model.ending_idx)
                
                rules[mask, pst + 1] = sample.squeeze(-1)[mask]
                log_probabilities[mask, pst + 1] = log_probability.squeeze(-1)[mask]

            length = (rules != model.ending_idx).sum(-1).unsqueeze(-1) - 1
            formatted_rules = torch.cat([length, rules], dim=1)

            log_probabilities = log_probabilities.sum(-1)

            formatted_rules = formatted_rules.data.cpu().numpy().tolist()
            log_probabilities = log_probabilities.data.cpu().numpy().tolist()
            for k in range(num_samples):
                length = formatted_rules[k][0]
                formatted_rules[k] = formatted_rules[k][1: 2 + length] + [log_probabilities[k]]

            rule_set = set([tuple(rule) for rule in formatted_rules])
            formatted_rules = [list(rule) for rule in rule_set]

            all_rules += formatted_rules

        return all_rules
    
    def load(self, checkpoint):
        """
        Load a checkpoint from file.
        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
        """
        if comm.get_rank() == 0:
            logging.info("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state["model"])

    def save(self, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """
        if comm.get_rank() == 0:
            logging.info("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = {
            "model": self.model.state_dict()
        }
        torch.save(state, checkpoint)    
