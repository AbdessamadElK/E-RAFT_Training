from __future__ import print_function, division
import sys

from pathlib import Path

from torch.utils.data import DataLoader

from loader.loader_dsec import DatasetProvider
from utils.dsec_utils import RepresentationType

# sys.path.append('core')

import json

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from model.eraft import ERAFT
# import evaluate
# import datasets

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# Global constants
CONFIG_PATH = "./dsec_standard.json"
DSEC_PATH = Path("/home/abdou/DSEC")


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(config, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    if config["stage"] == "dsec":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, config["lr"], config["num_steps"]+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(config):
    train_config = config["train"]
    n_first_channels = config["data_loader"]["train"]["n_first_channels"] 

    model = nn.DataParallel(ERAFT(config, n_first_channels), device_ids=config["train"]["gpus"])
    print("Parameter Count: %d" % count_parameters(model))

    if train_config["restore_ckpt"] is not None:
        model.load_state_dict(torch.load(train_config["restore_ckpt"]), strict=False)

    model.cuda()
    model.train()

    if config["stage"] != 'chairs':
        model.module.freeze_bn()

    provider = DatasetProvider(config["path"], mode = "train", representation_type=RepresentationType.VOXEL)

    train_loader = DataLoader(provider.get_dataset())

    # TODO: Implement fetch_dataloader function
    #train_loader = datasets.fetch_dataloader(config)

    optimizer, scheduler = fetch_optimizer(train_config, model)

    total_steps = 0
    scaler = GradScaler(enabled=train_config["mixed_precision"])
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            volume1 = data_blob["event_volume_old"].cuda()
            volume2 = data_blob["event_volume_new"].cuda()
            flow = data_blob["flow_gt"].cuda()
            valid = data_blob["flow_valid"].cuda()

            # Must be tested with cuda

            # if config.add_noise:
            #     stdv = np.random.uniform(0.0, 5.0)
            #     image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
            #     image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(volume1, volume2, iters=train_config["iters"])            

            loss, metrics = sequence_loss(flow_predictions, flow, valid, train_config["gamma"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["clip"])
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, config["name"])
                torch.save(model.state_dict(), PATH)

                # results = {}
                # for val_dataset in config.validation:
                #     if val_dataset == 'chairs':
                #         results.update(evaluate.validate_chairs(model.module))
                #     elif val_dataset == 'sintel':
                #         results.update(evaluate.validate_sintel(model.module))
                #     elif val_dataset == 'kitti':
                #         results.update(evaluate.validate_kitti(model.module))

                # logger.write_dict(results)
                
                model.train()
                if config["stage"] != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > config["num_steps"]:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % config["name"]
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name', default='raft', help="name your experiment")
    # parser.add_argument('--stage', help="determines which dataset to use for training") 
    # parser.add_argument('--restore_ckpt', help="restore checkpoint")
    # parser.add_argument('--small', action='store_true', help='use small model')
    # parser.add_argument('--validation', type=str, nargs='+')

    # parser.add_argument('--lr', type=float, default=0.00002)
    # parser.add_argument('--num_steps', type=int, default=100000)
    # parser.add_argument('--batch_size', type=int, default=6)
    # parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    # parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # parser.add_argument('--iters', type=int, default=12)
    # parser.add_argument('--wdecay', type=float, default=.00005)
    # parser.add_argument('--epsilon', type=float, default=1e-8)
    # parser.add_argument('--clip', type=float, default=1.0)
    # parser.add_argument('--dropout', type=float, default=0.0)
    # parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    # parser.add_argument('--add_noise', action='store_true')
    # args = parser.parse_args()

    config = json.load(open())

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(config)