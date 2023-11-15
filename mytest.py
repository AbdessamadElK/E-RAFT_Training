from pathlib import Path

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from loader.loader_dsec import Sequence
from loader.loader_dsec import DatasetProvider
from utils.dsec_utils import RepresentationType
from utils.visualization import visualize_optical_flow
from model.eraft import ERAFT

import skvideo.io
import skimage.io

from tqdm import tqdm

import numpy as np

dsec_path = Path("C:/users/public/dsec_flow")


provider = DatasetProvider(dsec_path, RepresentationType.VOXEL, mode = "train")

sequence_names = provider.name_mapper

loader = DataLoader(provider.get_dataset(), batch_size= 1, shuffle=False)

for data in loader:

    flow = data["flow_gt"].squeeze()
    flip = transforms.RandomHorizontalFlip(p = 1)

    flow_fl = flip(flow)
    flow_fl[0::] = - flow_fl[0::]

    flow_img, _ = visualize_optical_flow(flow.numpy(), return_bgr=True)
    flow_img = flow_img * 255

    flow_fl_img, _ = visualize_optical_flow(flow_fl.numpy(), return_bgr=True)
    flow_fl_img = flow_fl_img * 255

    vis = np.hstack([flow_img, flow_fl_img])

    skimage.io.imsave("C:/users/public/flow.png", vis.astype("uint8"))



    # c_sequence = torch.concatenate([sequence1, sequence2], dim = 1)
    break

quit()


config = json.load(open("./config/dsec_standard.json"))
n_first_channels = config["data_loader"]["train"]["args"]["num_voxel_bins"]

model = ERAFT(config, n_first_channels)

if torch.cuda.is_available():
    model = nn.DataParallel(model, device_ids=config["train"]["gpus"])


for i, batch in enumerate(loader):
    events_0 = batch["event_volume_old"]
    events_1 = batch["event_volume_new"]

    flow_n, predictions = model.forward(events_0, events_1)

    print(flow_n.shape)
    print(predictions[0].shape)
    break
