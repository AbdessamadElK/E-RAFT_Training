from pathlib import Path
from torch.utils.data import DataLoader

from argparse import ArgumentParser

import platform

import numpy as np

import torch

import cv2

import skvideo.io

from tqdm import tqdm

from utils.visualization import visualize_optical_flow
from utils.visualization import events_to_event_image
from utils.image_utils import forward_interpolate_pytorch

from loader.loader_dsec import DatasetProvider
from utils.dsec_utils import RepresentationType

def data_to_video(args):
    crop_size = (288, 384) if args.crop else None

    provider = DatasetProvider(Path(args.input), mode = "train",
                            representation_type=RepresentationType.VOXEL,
                            load_raw_events=args.events,
                            load_img=args.images,
                            crop_size=crop_size,
                            hflip=args.horizontal_flip,
                            vflip=args.vertical_flip)

    train_loader = DataLoader(provider.get_dataset())


    # Visualization
    VIDEO_FILE_TEMPLATE = "{}_{}.mp4"

    sequence_names = provider.name_mapper
    vis_count = 0

    # Get the video layout based on the number of sections that are to be displayed on each frame
    # One section for optical flow (always displayed)
    # Two sections for event sub-sequences
    # One section for image data
    imgs_per_frame = 1 + 2 * args.events + args.images
    # data_label = "flow" + "_events" * args.events + "_images" * args.images
    data_label = "video" # for now until I visualize the whole dataset

    layout = "row" if imgs_per_frame <= 3 else "matrix"

    for idx, name in enumerate(sequence_names):
        if not args.all:
            if args.sequence and args.sequence not in name:
                continue
            
            if args.number and vis_count >= args.number:
                continue

        loader_instance = train_loader.dataset.datasets[idx]
        
        # Create a video writer instance
        savepath = Path(args.output) / name

        if not savepath.is_dir():
            savepath.mkdir(parents = True, exist_ok = True)

        savepath = savepath / VIDEO_FILE_TEMPLATE.format(name, data_label)

        if savepath.is_file() and args.ignore_existing:
            continue

        # writer = skvideo.io.FFmpegWriter(savepath, outputdict = {'-r':str(3)})
        if platform.system() == "Windows":
            writer = skvideo.io.FFmpegWriter(str(savepath), outputdict={"-pix_fmt": "yuv420p"})
        else:
            writer = skvideo.io.FFmpegWriter(str(savepath))

        description = f"Sequence {vis_count+1}: {name}"

        for sample in tqdm(iter(loader_instance), total = len(loader_instance), desc = description):
            
            # Get events as image
            height, width = loader_instance.getHeightAndWidth()
            
            top_row_content = []
            bottom_row_content = []

            if args.events:
                for key in ["raw_events_old", "raw_events_new"]:
                    event_sequence = sample[key]
                    event_img = events_to_event_image(event_sequence, height, width)                    
                    event_img = event_img.numpy().transpose(1, 2, 0)
                    top_row_content.append(event_img)
            

            # Get optical flow as image
            flow_gt = sample["flow_gt"]

            if args.interpolate:
                # TODO : Find a way to interpolate optical flow and get a dense visualization
                pass

            flow_img, _ = visualize_optical_flow(flow_gt.squeeze(), return_bgr=True)
            flow_img = flow_img * 255

            ## Adding text label to the image (experimental)
            # labeled = cv2.putText(flow_img, "Optical Flow", (50, 50),
            #                               fontFace = 3,
            #                               fontScale = 1,
            #                               color = (255, 35, 35),
            #                               thickness=1)
            
            # flow_img = np.asarray(labeled)

            if layout == "row":
                top_row_content.append(flow_img)
            else:
                bottom_row_content.append(flow_img)

            # Get image data
            if args.images:
                image = sample["image"]

                if layout == "row":
                    top_row_content.append(image)
                else:
                    bottom_row_content.append(image)

            image_top_row = np.hstack(top_row_content)

            if layout == "matrix":
                image_bottom_row = np.hstack(bottom_row_content)
                sample_img = np.vstack([image_top_row, image_bottom_row])
            else:
                sample_img = image_top_row

            writer.writeFrame(sample_img.astype('uint8'))

        writer.close()

        vis_count += 1

#def data_to_video(args):
#   pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Dataset directory path")
    parser.add_argument("-o", "--output", type=str, help="Output directory path")

    parser.add_argument("-a", "--all", action="store_true", help="Visualize all sequences")
    parser.add_argument("-s", "--sequence", type=str, default="", help="Search query for sequences")
    parser.add_argument("-n", "--number", type=int, default=0, help="Number of sequences to visualize (0 means no limit)")

    # parser.add_argument("-f", "--flow", type=bool, default=True, help="ptical flow")
    parser.add_argument("-e", "--events", action="store_true", help="Include event data")
    parser.add_argument("-m", "--images", action="store_true", help="Include image data")

    parser.add_argument("-p", "--interpolate", action="store_true", help="Interpolate optical flow")
    parser.add_argument("-x", "--ignore_existing", action="store_true", help="Ignore sequences that are already visualized in the output directory using the same configuration")

    # transforms
    parser.add_argument("-c", "--crop", action="store_true", help = "Activate random cropping to (288, 384)")
    parser.add_argument("-h", "--horizontal_flip", action="store_true", help = "Activate random horizontal flipping")
    parser.add_argument("-v", "--vertical_flip", action="store_true", help = "Activate random vertical flipping")


    args = parser.parse_args()

    data_to_video(args)

    pass