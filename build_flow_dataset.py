from argparse import ArgumentParser

from pathlib import Path

import shutil
import os

from tqdm import tqdm

def copy_item(src:Path, dest:Path):
    """
    Copy either a file or a directory from src to dest
    src is the path to the source file/dir
    dest is the path to the destination file/dir (A new name may be used)
    """

    assert src.is_dir() or src.is_file()

    if src.is_file():
        shutil.copyfile(str(src), str(dest))
    
    if src.is_dir():
        if not dest.is_dir():
            dest.mkdir(parents=True, exist_ok=True)

        for item in src.iterdir():
            copy_item(item, dest / item.name)

    pass

def build(in_path : Path, out_path : Path, policy = "copy"):
    """
    Build DSEC Flow dataset from the Downloads directory built after downloading the whole dataset.
    We assume that the downloads directory has the following structure:
        Downloads
        |- train_events
            |- Sequence 1 (example : interlaken_00_a)
                |- events
                    |- left
                    |- right
            |- Sequence 2
            |- ...
        |- train_optical_flow
            |- Sequence 1
                |- flow
                    |- backward
                    |- forward
                    |- forward_timestamps.txt
                    |- backward_timestamps.txt
            |- Sequence 2
            |- ...
        |- train_images (optional)
            |- Sequence 1
                |- images
                    |- left
                    |- right
                    |- timestamps.txt
            | - ...
    
    The goal is to build DSEC_flow dataset which splits the data based on sequences and not the type of data,
    and only including sequences that have optical flow data available:
    
        DSEC_flow
        |- train
            |- zurich_city_01_a
                |- optical_flow_forward
                |- optical_flow backward
                |- events_left
                |- events_right
                |- images_left
                |- images_right
                |- flow_forward_timestamps.txt
                |- flow_backward_timestamps.txt
                |- images_timestamps.txt
            |- zurich_city_02_a
            |- ...
    """

    assert in_path.is_dir()

    policy = policy.lower()
    assert policy in ["move", "copy"]

    out_path = out_path / "train"

    if not out_path.is_dir():
        out_path.mkdir(parents = True, exist_ok = True)

    flow_dir = in_path / "train_optical_flow"
    
    if flow_dir.is_dir():
        flow_sequences = [x.name for x in flow_dir.iterdir()]
    else:
        flow_sequences = [x.name for x in out_path.iterdir()]
    
    for data_dir in in_path.iterdir():
        if data_dir.is_dir():
            confirmation = input(data_dir + " already exists, would you like to overwrite it? [y/N]")
            if confirmation.lower() in ["y", "yes"]:
                print("Warning :", data_dir.name, "will be overwriteen. Conflicts might result from this.")
            else:
                print("Ignoring", data_dir.name)
                continue

        oper = "Copying" if policy == "copy" else "Moving"
        description = oper + " " + data_dir.name

        for sequence_dir in tqdm(data_dir.iterdir(), desc=description):
            if not sequence_dir.name in flow_sequences or sequence_dir.is_file():
                continue

            for sub_dir in sequence_dir.iterdir():
                if sub_dir.is_file():
                    continue

                for item in sub_dir.iterdir():
                    name = "_".join([sub_dir.name, item.name])
                    destination = out_path / sequence_dir.name / name

                    copy_item(item, destination)                  

                if policy == "move":
                    shutil.rmtree(sub_dir)
    
    shutil.rmtree(in_path)

    return str(out_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path", type = str, help = "Downloaded dataset's directory path")
    parser.add_argument("-o", "--output_path", type = str, help= "DSEC-Flow output path")
    parser.add_argument("-p", "--policy", type = str, default="copy", help= "Construction policy : Copy or Move")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)


    build(input_path, output_path, policy=args.policy)