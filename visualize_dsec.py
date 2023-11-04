from pathlib import Path
from torch.utils.data import DataLoader

from argparse import ArgumentParser

import platform

import numpy as np

import skvideo.io

from tqdm import tqdm

from utils.visualization import visualize_optical_flow
from utils.visualization import events_to_event_image

from loader.loader_dsec import DatasetProvider
from utils.dsec_utils import RepresentationType

def data_to_video(args):

    provider = DatasetProvider(Path(args.input), mode = "train",
                            representation_type=RepresentationType.VOXEL,
                            load_raw_events=args.events,
                            load_img=args.images)

    train_loader = DataLoader(provider.get_dataset())


    # Visualization
    VIDEO_FILE_TEMPLATE = "{}_video.mp4"

    sequence_names = provider.name_mapper
    vis_count = 0

    # Get the video layout based on the number of sections that are to be displayed on each frame
    # One section for optical flow (always displayed)
    # Two sections for event sub-sequences
    # One section for image data
    imgs_per_frame = 1 + 2 * args.events + args.images

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

        savepath = savepath / VIDEO_FILE_TEMPLATE.format(name)

        # writer = skvideo.io.FFmpegWriter(savepath, outputdict = {'-r':str(3)})
        if platform.system() == "Windows":
            writer = skvideo.io.FFmpegWriter(str(savepath), outputdict={"-pix_fmt": "yuv420p"})
        else:
            writer = skvideo.io.FFmpegWriter(str(savepath))


        print(f"Sequence {vis_count+1}: {name}")


        for sample in tqdm(iter(loader_instance)):
            ts_start, ts_end = sample["timestamp"].squeeze()

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
                pass
            

            # Get optical flow as image
            flow_gt = sample["flow_gt"]
            flow_img, _ = visualize_optical_flow(flow_gt, return_bgr=True)
            flow_img = flow_img * 255

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

    args = parser.parse_args()

    data_to_video(args)

    pass