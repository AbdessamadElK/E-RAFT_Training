from pathlib import Path

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from loader.loader_dsec import Sequence
from loader.loader_dsec import DatasetProvider
from utils.dsec_utils import RepresentationType
from utils.visualization import visualize_optical_flow, events_to_event_image
from model.eraft import ERAFT

import skvideo.io
import skimage.io

from tqdm import tqdm

import numpy as np

dsec_path = Path("C:/users/public/dsec_flow")
save_path = Path("C:/users/public/test_flip")
if not save_path.is_dir():
    save_path.mkdir(parents=True, exist_ok=True)

CROP = False
HFLIP = False
VFLIP = False

CROP_SIZE = (288, 384) if CROP else None
flip = "vertical"
assert flip in {"horizontal", "vertical"}

provider = DatasetProvider(dsec_path, RepresentationType.VOXEL, mode = "train", load_raw_events=True, hflip=HFLIP, vflip=VFLIP, crop_size = CROP_SIZE, type="warm_start")

# flip_provider = DatasetProvider(dsec_path, RepresentationType.VOXEL, mode = "train", load_raw_events=True, hflip=flip=="horizontal", vflip=flip=="vertical", crop_size = None)

sequence_names = provider.name_mapper

loader = DataLoader(provider.get_dataset(), batch_size= 1, shuffle=False)
# flip_loader = DataLoader(flip_provider.get_dataset(), batch_size=1, shuffle=False)

dataset = provider.get_dataset()
# flip_dataset = flip_provider.get_dataset()

height, width = dataset.datasets[0].get_image_width_height()

# for i in range(len(dataset)):
#     data = dataset.__getitem__(i)
#     data_flip = flip_dataset.__getitem__(i)

#     events = data["raw_events_old"]
#     events_flip = data_flip["raw_events_old"]

#     img = events_to_event_image(events.squeeze(), height, width)
#     img_flip = events_to_event_image(events_flip.squeeze(), height, width)

#     if flip == "vertical":
#         img_flip = v2.RandomVerticalFlip(p=1)(img_flip)
#     else:
#         img_flip = v2.RandomHorizontalFlip(p=1)(img_flip)
        
#     img = img.numpy().transpose(1, 2, 0)
#     img_flip = img_flip.numpy().transpose(1, 2, 0)

#     loss_img = img - img_flip
#     loss = loss_img.astype(float).mean()

#     print(loss)

#     vis = np.hstack([img, img_flip, loss_img])
#     skimage.io.imsave(str(save_path / f"vis_{i}.png"), vis.astype("uint8"))

shape = None
for idx, data in tqdm(enumerate(loader), total = len(loader)):
    print(type(data))
    break        

    # Optical flow image
    flow_img, _ = visualize_optical_flow(data["flow_gt"].squeeze().numpy(), return_bgr=True)
    flow_img = flow_img * 255
    height, width, _ = flow_img.shape

    # Events as image
    event_sequence = data["raw_events_old"]
    event_img = events_to_event_image(event_sequence.squeeze().numpy(), height, width)
    event_img = event_img.numpy().transpose(1, 2, 0)

    vis = np.hstack([event_img, flow_img])

    skimage.io.imsave(str(save_path / f"vis_{idx}.png"), vis.astype("uint8"))

    # c_sequence = torch.concatenate([sequence1, sequence2], dim = 1)

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
