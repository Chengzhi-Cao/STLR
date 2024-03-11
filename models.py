import sys
import os
import os.path as osp
import logging
import argparse
import random
import json
from easydict import EasyDict
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader


from dataset import TrainDataset, ValidDataset, TestDataset, RuleDataset
from reasoning_evaluator import Reasoning_Evaluator
from generators import Generator
from utils import load_config, save_config, set_logger, set_seed
from trainer import Train_Evaluator, Train_Generator
import comm




def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', type=str)
    return parser.parse_args(args)

def main(args):
    cfgs = load_config(args.config)
    cfg = cfgs[0]

    if cfg.save_path is None:
        cfg.save_path = os.path.join('../outputs', datetime.now().strftime('%Y%m-%d%H-%M%S'))
    
    if cfg.save_path and not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    
    save_config(cfg, cfg.save_path)
    set_logger(cfg.save_path)
    set_seed(cfg.seed)


    train_set = TrainDataset()
    valid_set = ValidDataset()
    test_set = TestDataset()
    dataset = RuleDataset()

    if comm.get_rank() == 0:
        logging.info('-------------------------')
        logging.info('| Pre-train Generator')
        logging.info('-------------------------')
    generator = Generator(**cfg.generator.model)
    solver_g = Train_Generator(generator, gpu=cfg.generator.gpu)
    solver_g.train(dataset, **cfg.generator.pre_train)

    replay_buffer = list()
    for k in range(cfg.EM.num_iters):
        if comm.get_rank() == 0:
            logging.info('-------------------------')
            logging.info('| EM Iteration: {}/{}'.format(k + 1, cfg.EM.num_iters))
            logging.info('-------------------------')
        sampled_rules = solver_g.sample(cfg.EM.num_rules, cfg.EM.max_length)
        prior = [rule[-1] for rule in sampled_rules]
        rules = [rule[0:-1] for rule in sampled_rules]
        predictor = Reasoning_Evaluator(**cfg.predictor.model)
        predictor.set_rules(rules)
        optim = torch.optim.Adam(predictor.parameters(), **cfg.predictor.optimizer)
        solver_p = Train_Evaluator(predictor, train_set, valid_set, test_set, optim, gpus=cfg.predictor.gpus)
        solver_p.train(**cfg.predictor.train)
        likelihood = solver_p.compute_H(**cfg.predictor.H_score)
        posterior = [l + p * cfg.EM.prior_weight for l, p in zip(likelihood, prior)]
        for i in range(len(rules)):
            rules[i].append(posterior[i])
        replay_buffer += rules
        dataset = RuleDataset()
        solver_g.train(dataset, **cfg.generator.train)
        


    if comm.get_rank() == 0:
        logging.info('-------------------------')
        logging.info('| Best Rules')
        logging.info('-------------------------')
    


if __name__ == '__main__':
    main(parse_args())