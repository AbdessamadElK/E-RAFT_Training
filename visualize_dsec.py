from pathlib import Path
from torch.utils.data import DataLoader

import platform

import numpy as np

import skimage
import skvideo.io

from tqdm import tqdm

from utils.visualization import DsecFlowVisualizer
from utils.visualization import visualize_optical_flow
from utils.visualization import events_to_event_image

from loader.loader_dsec import DatasetProvider
from utils.dsec_utils import RepresentationType

# Data loading
dsec_path = Path("C:/users/public/DSEC")
provider = DatasetProvider(dsec_path, mode = "train", representation_type=RepresentationType.VOXEL)
train_loader = DataLoader(provider.get_dataset())


# Visualization
FILE_TEMPLATE = "sample_{}.png"
VIDEO_FILE_TEMPLATE = "{}_video.mp4"

sequence_names = provider.name_mapper

for idx, name in enumerate(sequence_names):
    loader_instance = train_loader.dataset.datasets[idx]
    slicer = loader_instance.event_slicer
    
    # Create a video writer instance
    savepath = dsec_path / "visulaization" / name

    if not savepath.is_dir():
            savepath.mkdir(parents = True, exist_ok = True)

    savepath = savepath / VIDEO_FILE_TEMPLATE.format(name)

    # writer = skvideo.io.FFmpegWriter(savepath, outputdict = {'-r':str(3)})
    if platform.system() == "Windows":
        writer = skvideo.io.FFmpegWriter(str(savepath), outputdict={"-pix_fmt": "yuv420p"})
    else:
        writer = skvideo.io.FFmpegWriter(str(savepath))


    print(f"Sequence {idx+1}: {name}")


    for sample in tqdm(iter(loader_instance)):
        ts_start, ts_end = sample["timestamp"].squeeze()

        # Get optical flow image
        flow_gt = sample["flow_gt"]
        flow_img, _ = visualize_optical_flow(flow_gt)
        flow_img = flow_img * 255

        # Get events as image
        slicer = loader_instance.event_slicer
        height, width = loader_instance.getHeightAndWidth()
        events = slicer.get_events(ts_start, ts_end)
        t = events["t"]
        x = events["x"]
        y = events["y"]
        p = events["p"]

        p = p * 2.0 - 1.0
        
        event_sequence = np.vstack([t, x, y, p]).transpose()
        event_img = events_to_event_image(event_sequence, height, width)
        event_img = event_img.numpy().transpose(1, 2, 0)

        sample_img = np.hstack([flow_img, event_img])

        # print(f"Writing {savepath}")
        # skimage.io.imsave(savepath, sample_img.astype('uint8'))
        writer.writeFrame(sample_img.astype('uint8'))

writer.close()