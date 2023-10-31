import math
from pathlib import Path
from typing import Dict, Tuple
import weakref

import cv2
import h5py
from numba import jit
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import visualization as visu
from matplotlib import pyplot as plt
from utils import transformers
import os
import imageio

from utils.dsec_utils import RepresentationType, VoxelGrid, flow_16bit_to_float

# To read HDF5 files on windows
import platform
if platform.system() == "Windows":
    import hdf5plugin

VISU_INDEX = 1

class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        self.t_offset = int(h5f['t_offset'][()])
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]


class Sequence(Dataset):
    def __init__(self, seq_path: Path, representation_type: RepresentationType, mode: str='test', delta_t_ms: 'int|None' = None,
                 num_bins: int=15, transforms=None, name_idx=0, visualize=False, load_imgs = False, load_raw_events = False):
        assert num_bins >= 1
        assert seq_path.is_dir()
        assert mode in {'train', 'test'}
        if mode == "test" and delta_t_ms is None:
            delta_t_ms = 100
        
        # For now only test with delta_t_ms = 100 ms
        if mode == "test" : assert delta_t_ms == 100

        # Save delta timestamp in micro-seconds
        if delta_t_ms is not None:
            self.delta_t_us = delta_t_ms * 1000

        '''
        Test directory Structure:

        Dataset
        └── test
        │   ├── interlaken_00_b
        │   │   ├── events_left
        │   │   │   ├── events.h5
        │   │   │   └── rectify_map.h5
        │   │   ├── image_timestamps.txt
        │   │   └── test_forward_flow_timestamps.csv
        │   ...
        └── train
        │   ├── zurich_city_01_a
        │   │   ├── events_left
        │   │   │   ├── events.h5
        │   │   │   └── rectify_map.h5
        │   │   ├── events_right
        │   │   ├── flow_forward
        │   │   ├── flow_backward
        │   │   ├── images_left
        │   │   ├── images_right
        │   │   ├── flow_forward_timestamps.txt
        │   │   ├── flow_backward_timestamps.txt
        │   │   └── images_timestamps.txt
        ... ...
        '''

        self.mode = mode
        self.name_idx = name_idx
        self.visualize_samples = visualize
        self.load_imgs = load_imgs
        self.load_raw_events = load_raw_events

        # Get timestamps files
        images_timestamp_path = seq_path / 'images_timestamps.txt'

        if self.mode == "test":
            flow_timestamp_path = seq_path / 'test_forward_flow_timestamps.csv'
        elif self.mode == "train":
            flow_timestamp_path = seq_path / 'flow_forward_timestamps.txt'
        
        assert flow_timestamp_path.is_file()
        assert images_timestamp_path.is_file()
        
        flow_timestamps = np.genfromtxt(
            flow_timestamp_path,
            delimiter=','
        )

        if self.mode == "test":
            self.idx_to_visualize = flow_timestamps[:,2]

        self.timestamps_images = np.loadtxt(images_timestamp_path, dtype="int64")

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        # Just for now, we always train with num_bins=15
        assert self.num_bins==15

        # Set event representation
        self.voxel_grid = None
        if representation_type == RepresentationType.VOXEL:
            self.voxel_grid = VoxelGrid((self.num_bins, self.height, self.width), normalize=True)

        #Load and compute timestamps and indices
        if self.mode == "test":
            # timestamps_images = np.loadtxt(seq_path / 'image_timestamps.txt', dtype='int64')
            image_indices = np.arange(len(self.timestamps_images))
            # But only use every second one because we train at 10 Hz, and we leave away the 1st & last one
            self.timestamps_flow = self.timestamps_images[::2][1:-1]
            self.indices = image_indices[::2][1:-1]

        elif self.mode == "train":
            self.timestamps_flow = flow_timestamps
            image_indices = [item[0] for item in enumerate(self.timestamps_images) if item[1] in self.timestamps_flow[:,0]]

        # Left events only
        ev_dir_location = seq_path / 'events_left'
        ev_data_file = ev_dir_location / 'events.h5'
        ev_rect_file = ev_dir_location / 'rectify_map.h5'

        h5f_location = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f_location
        self.event_slicer = EventSlicer(h5f_location)
        with h5py.File(str(ev_rect_file), 'r') as h5_rect:
            self.rectify_ev_map = h5_rect['rectify_map'][()]

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        # Localize flow files
        flow_dir = Path(seq_path / 'flow_forward')
        assert flow_dir.is_dir()
        self.flow_file_paths = sorted(flow_dir.iterdir())

        # Localize image files
        images_dir = Path(seq_path / 'images_left' / 'rectified')
        assert images_dir.is_dir()
        image_file_names = [Path(uri).name for uri in self.flow_file_paths]
        self.images_file_paths = [str(images_dir / name) for name in image_file_names]

    def events_to_voxel_grid(self, p, t, x, y, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        event_data_torch = {
            'p': torch.from_numpy(pol),
            't': torch.from_numpy(t),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
        }
        return self.voxel_grid.convert(event_data_torch)

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32')/256

    @staticmethod
    def load_flow(flowfile: Path):
        assert flowfile.exists()
        assert flowfile.suffix == '.png'
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def get_image_width_height(self):
        return self.height, self.width

    def __len__(self):
        return len(self.timestamps_flow)

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        # assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_map
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def get_data_sample(self, index, crop_window=None, flip=None):
        # First entry corresponds to all events BEFORE the flow map
        # Second entry corresponds to all events AFTER the flow map (corresponding to the actual fwd flow)
        volume_names = ['event_volume_old', 'event_volume_new']
        events_names = ['events_raw_old', 'events_raw_new']
        images_names = ['first_image', 'second_image']
        output = dict()

        assert index < len(self.timestamps_flow)

        if self.mode == "test":
            # Start and End times of the flow subsequences
            ts_start = [self.timestamps_flow[index] - self.delta_t_us, self.timestamps_flow[index]]
            ts_end = [self.timestamps_flow[index], self.timestamps_flow[index] + self.delta_t_us]

            file_index = self.indices[index]
            output['file_index'] = file_index
            output['timestamp'] = self.timestamps_flow[index]

            # Save sample for benchmark submission
            output['save_submission'] = file_index in self.idx_to_visualize
            output['visualize'] = self.visualize_samples
        
        elif self.mode == "train":
            # Start and End times of the flow subsequences
            t0, t1 = self.timestamps_flow[index]
            ts_start = [t0, (t0+t1)//2]
            ts_end = [(t0+t1)//2, t1]

            # Timestamp
            output['timestamp'] = self.timestamps_flow[index]


        for i in range(len(volume_names)):
            event_data = self.event_slicer.get_events(ts_start[i], ts_end[i])

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            xy_rect = self.rectify_events(x, y)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            if crop_window is not None:
                # Cropping (+- 2 for safety reasons)
                x_mask = (x_rect >= crop_window['start_x']-2) & (x_rect < crop_window['start_x']+crop_window['crop_width']+2)
                y_mask = (y_rect >= crop_window['start_y']-2) & (y_rect < crop_window['start_y']+crop_window['crop_height']+2)
                mask_combined = x_mask & y_mask
                p = p[mask_combined]
                t = t[mask_combined]
                x_rect = x_rect[mask_combined]
                y_rect = y_rect[mask_combined]

            if self.load_raw_events:
                p = p * 2.0 - 1.0
                event_sequence = np.vstack([t, x_rect, y_rect, p]).transpose()
                output[events_names[i]] = event_sequence

            if self.voxel_grid is None:
                raise NotImplementedError
            else:
                event_representation = self.events_to_voxel_grid(p, t, x_rect, y_rect)
                output[volume_names[i]] = event_representation

            if self.load_imgs:
                
                pass

            output['name_map']=self.name_idx

        # Also include optical flow ground trugh when training
        flow_path = Path(self.flow_file_paths[index])
        output['flow_gt'], output['flow_valid'] = self.load_flow(flow_path)
        
        # Channels first
        output['flow_gt'] = output['flow_gt'].transpose(2, 0, 1)

        return output

    def __getitem__(self, idx):
        try:
            sample =  self.get_data_sample(idx)
        except AssertionError:
            raise StopIteration
        return sample


class SequenceRecurrent(Sequence):
    def __init__(self, seq_path: Path, representation_type: RepresentationType, mode: str='test', delta_t_ms: int=100,
                 num_bins: int=15, transforms=None, sequence_length=1, name_idx=0, visualize=False):
        super(SequenceRecurrent, self).__init__(seq_path, representation_type, mode, delta_t_ms, transforms=transforms,
                                                name_idx=name_idx, visualize=visualize)
        self.sequence_length = sequence_length
        self.valid_indices = self.get_continuous_sequences()

    def get_continuous_sequences(self):
        continuous_seq_idcs = []
        if self.sequence_length > 1:
            for i in range(len(self.timestamps_flow)-self.sequence_length+1):
                diff = self.timestamps_flow[i+self.sequence_length-1] - self.timestamps_flow[i]
                if diff < np.max([100000 * (self.sequence_length-1) + 1000, 101000]):
                    continuous_seq_idcs.append(i)
        else:
            for i in range(len(self.timestamps_flow)-1):
                diff = self.timestamps_flow[i+1] - self.timestamps_flow[i]
                if diff < np.max([100000 * (self.sequence_length-1) + 1000, 101000]):
                    continuous_seq_idcs.append(i)
        return continuous_seq_idcs

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < len(self)

        # Valid index is the actual index we want to load, which guarantees a continuous sequence length
        valid_idx = self.valid_indices[idx]

        sequence = []
        j = valid_idx

        ts_cur = self.timestamps_flow[j]
        # Add first sample
        sample = self.get_data_sample(j)
        sequence.append(sample)

        # Data augmentation according to first sample
        crop_window = None
        flip = None
        if 'crop_window' in sample.keys():
            crop_window = sample['crop_window']
        if 'flipped' in sample.keys():
            flip = sample['flipped']

        for i in range(self.sequence_length-1):
            j += 1
            ts_old = ts_cur
            ts_cur = self.timestamps_flow[j]
            assert(ts_cur-ts_old < 100000 + 1000)
            sample = self.get_data_sample(j, crop_window=crop_window, flip=flip)
            sequence.append(sample)

        # Check if the current sample is the first sample of a continuous sequence
        if idx==0 or self.valid_indices[idx]-self.valid_indices[idx-1] != 1:
            sequence[0]['new_sequence'] = 1
            print("Timestamp {} is the first one of the next seq!".format(self.timestamps_flow[self.valid_indices[idx]]))
        else:
            sequence[0]['new_sequence'] = 0
        return sequence

class DatasetProvider:
    def __init__(self, dataset_path: Path, representation_type: RepresentationType, delta_t_ms: int=100, num_bins=15,
                 mode = 'test', type='standard', config=None, visualize=False):
        path = dataset_path / mode
        assert dataset_path.is_dir(), str(dataset_path)
        assert path.is_dir(), str(path)
        assert delta_t_ms == 100
        self.config=config
        self.name_mapper = []

        sequences = list()
        for child in path.iterdir():
            self.name_mapper.append(child.name)
            if type == 'standard':
                sequences.append(Sequence(child, representation_type, mode, delta_t_ms, num_bins,
                                               transforms=[],
                                               name_idx=len(self.name_mapper)-1,
                                               visualize=visualize))
            elif type == 'warm_start':
                sequences.append(SequenceRecurrent(child, representation_type, mode, delta_t_ms, num_bins,
                                                        transforms=[], sequence_length=1,
                                                        name_idx=len(self.name_mapper)-1,
                                                        visualize=visualize))
            else:
                raise Exception('Please provide a valid subtype [standard/warm_start] in config file!')

        self.dataset = torch.utils.data.ConcatDataset(sequences)

    def get_dataset(self):
        return self.dataset


    def get_name_mapping_test(self):
        return self.name_mapper

    def summary(self, logger):
        logger.write_line("================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__, True)
        logger.write_line("Number of Voxel Bins: {}".format(self.dataset.datasets[0].num_bins), True)
