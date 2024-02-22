import torch
from torch.utils.tensorboard import SummaryWriter

from utils.visualization import events_to_event_image, visualize_optical_flow

import numpy as np

from tqdm import tqdm

@torch.no_grad()
def evaluate_dsec(model, val_loader, val_step, iters = 12, writer : SummaryWriter = None):
    # Random visualization index
    vis_idx = np.random.randint(0, len(val_loader))

    # vis_idx = 10 # for debugging

    epe_list = []

    for idx, data in tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluating", leave=False):
        volume_1 = data["event_volume_old"].cuda()
        volume_2 = data["event_volume_new"].cuda()
        flow_gt = data["flow_gt"].cuda()
        valid = data["flow_valid"].cuda()

        valid = valid >= 0.5

        _, preds = model(volume_1, volume_2, iters)
        prediction = preds[-1]

        epe = torch.sum((prediction - flow_gt)**2, dim=1).sqrt()
        epe_list.append(epe.cpu().view(-1).numpy())

        if (not writer is None) and (idx == vis_idx):
            top_row_content = []
            bottom_row_content = []

            # Prediction image
            pred_img, _ = visualize_optical_flow(prediction.cpu().squeeze().numpy(), return_bgr=True)
            bottom_row_content.append(pred_img)

            # Ground truth optical flow image
            flow_img, _ = visualize_optical_flow(flow_gt.cpu().squeeze().numpy(), return_bgr=True)
            bottom_row_content.append(flow_img)
            height, width, _ = flow_img.shape

            # Image data
            image = data["image"].squeeze().numpy().transpose(1, 2, 0)
            top_row_content.append(image / 255)

            # Events as image
            event_sequence = data["raw_events_old"]
            event_img = events_to_event_image(event_sequence.squeeze().numpy(), height, width)
            event_img = event_img.numpy().transpose(1, 2, 0)
            top_row_content.append(event_img / 255)

            print("image :", image.shape)
            print("event_img :", event_img.shape)
            print("pred_img :", pred_img.shape)
            print("flow_img :", flow_img.shape)

            # Build visualization image
            image_top_row = np.hstack(top_row_content)
            image_bottom_row = np.hstack(bottom_row_content)
            image_matrix = np.vstack([image_top_row, image_bottom_row])

            # Visualize
            writer.add_image("Visualization", image_matrix, val_step, dataformats="HWC")

    
    epe_all = np.concatenate(epe_list)

    results = {
        'val_epe': np.mean(epe_all),
        'val_1px': np.mean(epe_all < 1),
        'val_3px': np.mean(epe_all < 3),
        'val_5px': np.mean(epe_all < 5),
    }

    return results