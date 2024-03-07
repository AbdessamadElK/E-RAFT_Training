import torch
import torch.nn as nn

from model.eraft import ERAFT

import numpy as np

from tqdm import tqdm
from tabulate import tabulate

import json
import csv

from argparse import ArgumentParser

from loader.loader_dsec import DatasetProvider
from utils.dsec_utils import RepresentationType

from torch.utils.data import DataLoader

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

@torch.no_grad()
def get_epe_results(epe_list, px_metrics:list = [1, 3, 5]):
    results = {}
    for epe in epe_list:
        if not "epe" in results:
            results["epe"] = 0

        # Accumulate the Average EPE for each estimate
        results["epe"] += epe.mean().item()

        for n in px_metrics:
            # Also inlclude npx metrics
            key = f'{n}px'
            if not key in results:
                results[key] = 0
            results[key] += (epe < n).float().mean().item()
    
    for key in results:
        # Calculate the mean of each metric
        results[key] = results[key] / len(epe_list)

    return results

MAX_FLOW  = 400

@torch.no_grad()
def evaluate_dsec(model, data_loader, iters = 12, individual = False, seq_names = None):

    epe_list = []
    epe_list_seq = []

    seq_idx = 0

    individual_results = [] if individual else None

    model.eval()
    for data in tqdm(data_loader, total=len(data_loader), desc="Evaluating", leave=False):
        volume_1 = data["event_volume_old"].cuda()
        volume_2 = data["event_volume_new"].cuda()
        flow_gt = data["flow_gt"]
        valid = data["flow_valid"]

        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        valid = (valid >= 0.5) & (mag < MAX_FLOW)

        _, preds = model(volume_1, volume_2, iters)
        prediction = preds[-1]

        epe = torch.sum((prediction.cpu() - flow_gt)**2, dim=1).sqrt()
        epe_list.append(epe.view(-1)[valid.view(-1)])

        if individual:
            if data["name_map"] != seq_idx:
                seq_results = get_epe_results(epe_list_seq)
                if seq_names is not None:
                    seq_results["seq_name"] = seq_names[seq_idx]
                else:
                    seq_results["seq_name"] = f"Sequence_{seq_idx + 1}"
                individual_results.append(seq_results)

                epe_list_seq = []
                seq_idx = data["name_map"]

            epe_list_seq.append(epe.cpu().view(-1)[valid.view(-1)])

    results = get_epe_results(epe_list)

    return results, individual_results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Saved model file (.pth)")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset directory path")
    parser.add_argument("-s", "--split", type=str, help="Data split to evaluate on [train/validation/test]")
    parser.add_argument("-c", "--config", type=str, help="Config file path")
    parser.add_argument("-n", "--num_iters", type=int, default=12, help="Number of iterations")
    parser.add_argument("-i", "--individual", action="store_true", help="Return results for each sequence")
    
    args = parser.parse_args()
    config = json.load(open(args.config))

    path = Path(args.dataset)
    assert path.is_dir()

    model_file = Path(args.model)
    assert model_file.is_file()

    split = args.split
    assert split in ["train", "validation", "test"]

    # Dataloader
    provider = DatasetProvider(path,
                               mode = split,
                               representation_type=RepresentationType.VOXEL,
                               crop_size=config["train"]["crop_size"],
                               hflip=config["train"]["horizontal_flip"],
                               vflip=config["train"]["vertical_flip"])
    
    data_loader = DataLoader(provider.get_dataset())

    # Model
    n_first_channels = config["data_loader"]["train"]["args"]["num_voxel_bins"]
    model = nn.DataParallel(ERAFT(config, n_first_channels), device_ids=config["train"]["gpus"])

    model.load_state_dict(torch.load(model_file), strict=False)

    model_name = model_file.name.split(".")[0]

    # Evaluation
    print(f'Evaluating "{model_name}" on the {split} split of DSEC Dataset:')
    results, individual_results = evaluate_dsec(model,
                                                data_loader,
                                                iters=args.num_iters,
                                                individual=args.individual,
                                                seq_names=provider.name_mapper)

    # Displaying results
    print("\nResults:\n\n")

    if not args.individual:
        for key in results:
            print(key, ":", results[key])
    else:
        # Savepath
        savepath = Path(f"./results")
        savepath.mkdir(parents = True, exist_ok = True)
        savepath = savepath / f"{model_name}_{split}_crop.csv"

        # Also add total results
        results["seq_name"] = "All"
        individual_results.insert(0, results)

        # Write CSV file
        LABELS = ["seq_name", "epe", "1px", "3px", "5px"]
        table = []
        with open(savepath, "w") as f:
            writer = csv.writer(f)
            writer.writerow(LABELS)
            for result in individual_results:
                items = sorted(result.items(), key = lambda it : LABELS.index(it[0]))
                values = [item[1] for item in items]
                table.append(values)
                writer.writerow(values)

        # Display results on the console
        print(tabulate(table, headers=LABELS))