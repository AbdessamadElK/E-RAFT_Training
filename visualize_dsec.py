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
dsec_path = Path("C:/users/public/DSEC_flow")

provider = DatasetProvider(dsec_path, mode = "train",
                           representation_type=RepresentationType.VOXEL,
                           load_raw_events=True,
                           load_img=True)

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

        # Get events as image
        height, width = loader_instance.getHeightAndWidth()
        
        event_images = []
        for key in ["raw_events_old", "raw_events_new"]:
            event_sequence = sample[key]
            event_img = events_to_event_image(event_sequence, height, width)
            event_img = event_img.numpy().transpose(1, 2, 0)
            event_images.append(event_img)

        image_top_row = np.hstack(event_images)

        # Get optical flow as image
        flow_gt = sample["flow_gt"]
        flow_img, _ = visualize_optical_flow(flow_gt, return_bgr=True)
        flow_img = flow_img * 255

        # Get image data
        image = sample["image"]

        image_bottom_row = np.hstack([image, flow_img])

        sample_img = np.vstack([image_top_row, image_bottom_row])

        # print(f"Writing {savepath}")
        # skimage.io.imsave(savepath, sample_img.astype('uint8'))
        writer.writeFrame(sample_img.astype('uint8'))

    if idx >= 3:
        # Only visualize four sequences for now
        break

writer.close()