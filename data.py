import numpy as np

import h5py

import utils

import torch

import xml.etree.ElementTree as etree

from glob import glob

from torch.utils.data import Dataset

from pathlib import Path

from typing import Sequence

def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)

class Sample:
    def __init__(self, fname: str):
        """
        Arguments:
            fname (string): Filename of the HDF5 file to load.
        """
        self.kspace: np.ndarray | torch.Tensor = None
        self.mask: np.ndarray | torch.Tensor = None
        self.metadata: dict = {}

        if fname:
            with h5py.File(fname) as file:
                if 'kspace' in file.keys():
                    self.kspace = file['kspace'][()]
                    if len(self.kspace.shape) == 3:
                        # single-coil data: (slices, height, width)
                        self.kspace = np.expand_dims(self.kspace, 1) # shape: (slices, 1, height, width)
                    else:
                        # multi-coil data: (slices, coils, height, width)
                        assert len(self.kspace.shape) == 4, f'Invalid shape for multi-coil k-space: {self.kspace.shape}'

                if 'mask' in file.keys():
                    self.mask = file['mask'][()]
                    assert self.mask.shape[0] == self.kspace.shape[-1], f'k-space mask shape ({self.mask.shape}) should match k-space width ({self.kspace.shape[-1]})'

                et_root = etree.fromstring(file["ismrmrd_header"][()])
    
                enc = ["encoding", "encodedSpace", "matrixSize"]
                enc_size = (
                    int(et_query(et_root, enc + ["x"])),
                    int(et_query(et_root, enc + ["y"])),
                    int(et_query(et_root, enc + ["z"])),
                )
                rec = ["encoding", "reconSpace", "matrixSize"]
                recon_size = (
                    int(et_query(et_root, rec + ["x"])),
                    int(et_query(et_root, rec + ["y"])),
                    int(et_query(et_root, rec + ["z"])),
                )
    
                lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
                enc_limits_center = int(et_query(et_root, lims + ["center"]))
                enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1
    
                padding_left = enc_size[1] // 2 - enc_limits_center
                padding_right = padding_left + enc_limits_max
    
                self.metadata = {
                    "padding_left": padding_left,
                    "padding_right": padding_right,
                    "encoding_size": enc_size,
                    "recon_size": recon_size,
                    **file.attrs,
                }

    def validate(self):
        assert len(self.kspace.shape) == 4, f'Invalid shape for k-space: {self.kspace.shape}'
        assert self.mask.shape[0] == self.kspace.shape[-1], f'k-space mask shape ({self.mask.shape}) should match k-space width ({self.kspace.shape[-1]})'
    
    def is_numpy(self) -> bool:
        return isinstance(self.kspace, np.ndarray)

    @staticmethod
    def from_numpy(kspace: np.ndarray, mask: np.ndarray = None, metadata: dict = None) -> 'Sample':
        sample = Sample(None)
        sample.kspace = kspace
        sample.mask = mask
        sample.metadata = metadata
        sample.validate()
        return sample
    
    @staticmethod
    def from_torch(kspace: torch.Tensor, mask: torch.Tensor = None, metadata: dict = None) -> 'Sample':
        sample = Sample(None)
        sample.kspace = kspace
        sample.mask = mask
        sample.metadata = metadata
        sample.validate()
        return sample
    
    def at_slice(self, idx: int) -> 'Sample':
        assert idx >= 0 and idx < self.num_slices, f'Invalid slice index: {idx}'

        sample = Sample(None)
        if isinstance(self.kspace, np.ndarray):
            sample.kspace = self.kspace[idx][None]
        else:
            sample.kspace = self.kspace[idx].unsqueeze(0)
        sample.mask = self.mask
        sample.metadata = self.metadata
        sample.validate()
        return sample

    @property
    def num_coils(self):
        return self.kspace.shape[1]

    @property
    def num_slices(self):
        return self.kspace.shape[0]

    @property
    def height(self):
        return self.kspace.shape[2]

    @property
    def width(self):
        return self.kspace.shape[3]
    
    @property
    def shape(self):
        return self.kspace.shape

    @property
    def image(self):
        return utils.kspace_to_image(self.kspace)
    
    @property
    def masked_image(self):
        return utils.kspace_to_image(self.masked_kspace)

    @property
    def masked_kspace(self):
        if self.mask is not None:
            return self.kspace * self.mask + 0.0
        else:
            return self.kspace

class MRIDataset(Dataset):
    def __init__(self, path: Path):
        """
        Arguments:
            path (Path): Path to the directory containing HDF5 files.
        """
        self.path: Path = path
        self.files: list = glob(f'{self.path}/*.h5')

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Sample:
        return Sample(self.files[idx])
