from fileinput import filename
import glob
import os
from itertools import chain

import h5py
import numpy as np
import time
import pickle
import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger
import os, psutil
from itertools import groupby
process = psutil.Process(os.getpid())

logger = get_logger('OwnHDF5LazyDataset')




class OwnLazyHDF5Dataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the npz files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, file_path,
                 phase,
                 transformer_config,
                 global_normalization=True):
        """
        :param file_path: paths to npz files containing raw data as well as labels and per pixel weights (optional)
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :param transformer_config: data augmentation configuration
        """
        assert phase in ['train', 'val', 'test']
        
        self.phase = phase
        self.file_path = file_path

        # load example to get stats and check weight==none
        raw, label, weight, stats=self.get_data(0)

        if global_normalization:
            stats = stats
        else:
            stats = {'pmin': None, 'pmax': None, 'mean': None, 'std': None}

        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()
        
        if phase != 'test':
            # create label/weight transform only in train/val phase
            
            self.label_transform = self.transformer.label_transform()

            if weight is not None and weight.size>1:
                # look for the weight map in the raw file
                self.weight_transform = self.transformer.weight_transform()
            else:
                self.weight_map = None

        self.patch_count = len(self.file_path)
        logger.info(f'Number of patches: {self.patch_count}')


    def get_data(self,index):
        data = np.load(self.file_path[index],allow_pickle=True)
        return (data["raw"], data["label"], data["weight"], data["stats"])

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw,label,weight,_=self.get_data(idx)
        # get the raw data patch for a given slice
        raw_patch_transformed = self.raw_transform(raw)
        #TODO
        if self.phase == 'test':
            pass
        #     # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
        #     if len(raw_idx) == 4:
        #         raw_idx = raw_idx[1:]
        #     return raw_patch_transformed, raw_idx
        else:
            # get the slice for a given index 'idx'
            
            label_patch_transformed = self.label_transform(label)
            if weight is not None and weight.size>1:
                weight_patch_transformed = self.weight_transform(weight)
                return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    def __len__(self):
        return self.patch_count

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load files to process
        file_paths = glob.glob(os.path.join(phase_config['temp_file_path'], "*.npz"))
        util_func = lambda x: x[:x.rfind('_')]
        temp = sorted(file_paths, key = util_func)
        res = [list(ele) for i, ele in groupby(temp, util_func)]
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths

        datasets = []
        for file_path in res:
            try:
                logger.info(f'Loading {phase} set from: {file_path}...')
                 
                dataset = cls(file_path=file_path,
                              phase=phase,
                              transformer_config=transformer_config,
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets





def create_slices_of_file(file_path,
                phase,
                slice_builder_config,
                transformer_config,
                temp_file_path,
                mirror_padding=(16, 32, 32),
                raw_internal_path='raw',
                label_internal_path='label',
                weight_internal_path=None,
                global_normalization=True):
    """
    generate npz file for each slice
    :param file_path: path to H5 file containing raw data as well as labels and per pixel weights (optional)
    :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
        only during the 'train' phase
    :para'/home/adrian/workspace/ilastik-datasets/VolkerDeconv/train'm slice_builder_config: configuration of the SliceBuilder
    :param transformer_config: data augmentation configuration
    :param mirror_padding (int or tuple): number of voxels padded to each axis
    :param raw_internal_path (str or list): H5 internal path to the raw dataset
    :param label_internal_path (str or list): H5 internal path to the label dataset
    :param weight_internal_path (str or list): H5 internal path to the per pixel weights
    """
    assert phase in ['train', 'val', 'test']
    if phase in ['train', 'val']:
        mirror_padding = None

    if mirror_padding is not None:
        if isinstance(mirror_padding, int):
            mirror_padding = (mirror_padding,) * 3
        else:
            assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"

    mirror_padding = mirror_padding
    phase = phase
    file_path = file_path

    input_file = create_h5_file(file_path)

    raw = fetch_and_check(input_file, raw_internal_path)

    if global_normalization:
        stats = calculate_stats(raw)
    else:
        stats = {'pmin': None, 'pmax': None, 'mean': None, 'std': None}

    transformer = transforms.Transformer(transformer_config, stats)

    if phase != 'test':
        # create label/weight transform only in train/val phase
        label = fetch_and_check(input_file, label_internal_path)

        if weight_internal_path is not None:
            # look for the weight map in the raw file
            weight_map = fetch_and_check(input_file, weight_internal_path)
        else:
            weight_map = None

        _check_volume_sizes(raw, label)
    else:
        # 'test' phase used only for predictions so ignore the label dataset
        label = None
        weight_map = None

        # add mirror padding if needed
        if mirror_padding is not None:
            z, y, x = mirror_padding
            pad_width = ((z, z), (y, y), (x, x))
            if raw.ndim == 4:
                channels = [np.pad(r, pad_width=pad_width, mode='reflect') for r in raw]
                raw = np.stack(channels)
            else:
                raw = np.pad(raw, pad_width=pad_width, mode='reflect')

    # build slice indices for raw and label data sets
    slice_builder = get_slice_builder(raw, label, weight_map, slice_builder_config)
    raw_slices = slice_builder.raw_slices
    label_slices = slice_builder.label_slices
    weight_slices = slice_builder.weight_slices
    patch_count = len(raw_slices)
    if weight_slices==None:
        weight_slices=[None]*len(raw_slices)
    file_name = os.path.basename(file_path)
    for i,(raw_slice,label_slice,weight_slice) in enumerate(zip(raw_slices,label_slices,weight_slices)):
        store_filename=temp_file_path+"/"+file_name.replace(".h5","_"+str(i))+'.npz'
        if weight_slice!=None:
            ws=weight_map[weight_slice]
        else:
            ws=None
        np.savez(store_filename, raw=raw[raw_slice], label=label[label_slice],weight=ws,stats=stats)


    
    logger.info(f'Number of patches: {patch_count}')

def fetch_and_check(input_file, internal_path):
    ds = input_file[internal_path][:]
    if ds.ndim == 2:
        # expand dims if 2d
        ds = np.expand_dims(ds, axis=0)
    return ds

def create_h5_file(file_path):
    return h5py.File(file_path, 'r')

def _check_volume_sizes(raw, label):
    def _volume_shape(volume):
        if volume.ndim == 3:
            return volume.shape
        return volume.shape[1:]

    assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
    assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

    assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'

def create_all_slices(dataset_config, phase):
    phase_config = dataset_config[phase]

    temp_file_path=phase_config["temp_file_path"]

    # load data augmentation configuration
    transformer_config = phase_config['transformer']
    # load slice builder config
    slice_builder_config = phase_config['slice_builder']
    # load files to process
    file_paths = phase_config['file_paths']
    # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
    # are going to be included in the final file_paths
    file_paths = traverse_h5_paths(file_paths)

    datasets = []
    for file_path in file_paths:
        try:
            logger.info(f'Loading {phase} set from: {file_path}...')
                

            create_slices_of_file(file_path=file_path,
                            phase=phase,
                            slice_builder_config=slice_builder_config,
                            transformer_config=transformer_config,
                            temp_file_path=temp_file_path,
                            mirror_padding=dataset_config.get('mirror_padding', None),
                            raw_internal_path=dataset_config.get('raw_internal_path', 'raw'),
                            label_internal_path=dataset_config.get('label_internal_path', 'label'),
                            weight_internal_path=dataset_config.get('weight_internal_path', None),
                            global_normalization=dataset_config.get('global_normalization', None))
        except Exception:
            logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
    return datasets

def traverse_h5_paths(file_paths):
    assert isinstance(file_paths, list)
    results = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            # if file path is a directory take all H5 files in that directory
            iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
            for fp in chain(*iters):
                results.append(fp)
        else:
            results.append(file_path)
    return results