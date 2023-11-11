import torch
from torch.utils.tensorboard import SummaryWriter

from utils.visualization import events_to_event_image, visualize_optical_flow

import numpy as np

from tqdm import tqdm

@torch.no_grad()
def evaluate_dsec(model, val_loader, val_step, iters = 12, writer : SummaryWriter = None):
    # Random visualization index
    vis_idx = np.random.randint(0, len(val_loader))

    # Visualize at the 10th step during debugging
    vis_idx = 10

    epe_list = []

    for idx, data in tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluating"):
        volume_1 = data["event_volume_old"]
        volume_2 = data["event_volume_new"]
        flow_gt = data["flow_gt"]
        valid = data["flow_valid"]

        valid = valid >= 0.5

        _, preds = model(volume_1, volume_2, iters)
        prediction = preds[-1].cpu()

        epe = torch.sum((prediction - flow_gt)**2, dim=1).sqrt()
        epe_list.append(epe.view(-1).numpy())

        if (not writer is None) and (idx == vis_idx):
            top_row_content = []
            bottom_row_content = []

            # Prediction image
            pred_img, _ = visualize_optical_flow(prediction.numpy().squeeze(), return_bgr=True)
            pred_img = pred_img * 255
            bottom_row_content.append(pred_img)

            # Ground truth optical flow image
            flow_img, _ = visualize_optical_flow(flow_gt.numpy().squeeze(), return_bgr=True)
            flow_img = flow_img * 255
            bottom_row_content.append(flow_img)
            height, width, _ = flow_img.shape

            # Image data
            image = data["image"]
            top_row_content.append(image)
            print(image.shape)

            # Events as image
            event_sequence = data["raw_events_old"]
            event_img = events_to_event_image(event_sequence.numpy().squeeze(), height, width)
            event_img = event_img.numpy().transpose(1, 2, 0)
            top_row_content.append(event_img)
            print(event_img.shape)

            # Visualize
            image_top_row = np.hstack(top_row_content)
            image_bottom_row = np.hstack(bottom_row_content)
            image_matrix = np.vstack([image_top_row, image_bottom_row])

            writer.add_image("Visualization", image_matrix, val_step)

    
    epe_all = np.concatenate(epe_list)

    results = {
        'val_epe': epe_all.mean().item(),
        'val_1px': (epe_all < 1).float().mean().item(),
        'val_3px': (epe_all < 3).float().mean().item(),
        'val_5px': (epe_all < 5).float().mean().item(),
    }

    return results