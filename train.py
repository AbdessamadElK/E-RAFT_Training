#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

from pathlib import Path

from torch.utils.data import DataLoader


from loader.loader_dsec import DatasetProvider
from utils.dsec_utils import RepresentationType

dsec_path = Path("/home/abdou/DSEC")

provider = DatasetProvider(dsec_path, mode = "train", representation_type=RepresentationType.VOXEL)

train_loader = DataLoader(provider.get_dataset())

for data in train_loader:
    print(data["timestamp"])