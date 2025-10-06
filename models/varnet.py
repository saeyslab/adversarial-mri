import torch

import utils

import numpy as np

from torchvision.transforms import CenterCrop

from pathlib import Path

from fastmri.models import VarNet as VN

from models.model import Model

from data import Sample

class VarNet(Model):
    def __init__(self, subset: str, coil: str, weight_path: Path, config: dict, **kwargs):
        super().__init__(weight_path, **kwargs)

        self.model = VN(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8).to(self.device).eval()

        if not weight_path.exists():
            print('[*] Downloading weights...')
            key = f'varnet-{subset}-{coil}'
            url = f'{config["varnet-url"]}/{config[key]}'
            utils.download_model(url, weight_path)
            print(f'[*] Weights saved to {weight_path}')
        else:
            print(f'[*] Loading weights from {weight_path}')
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model = torch.nn.DataParallel(self.model)

    def forward(self, sample: Sample, batch_size: int = 4) -> torch.Tensor:
        # Get masked kspace
        masked_kspace = torch.from_numpy(sample.masked_kspace).to(torch.complex64)  # (slices, coils, H, W)
        masked_kspace = torch.view_as_real(masked_kspace) # (slices, coils, H, W, 2)
    
        # Create mask
        shape = np.array(masked_kspace.shape)
        num_cols = shape[-2]
        shape[:-3] = 1
        mask_shape = [1] * len(shape)
        mask_shape[-2] = num_cols
        mask_torch = torch.from_numpy(sample.mask.reshape(*mask_shape).astype(np.float32))
        mask_torch = mask_torch.reshape(*mask_shape).bool().to(self.device)
    
        acq_start = sample.metadata["padding_left"]
        acq_end = sample.metadata["padding_right"]
        mask_torch[:, :, :acq_start] = 0
        mask_torch[:, :, acq_end:] = 0
    
        crop_size = (sample.metadata["recon_size"][0], sample.metadata["recon_size"][1])
        orig_size = (masked_kspace.shape[-3], masked_kspace.shape[-2])
    
        # Iterate over slices
        outputs = []
        for s in range(sample.num_slices):
            kspace_slice = masked_kspace[s].unsqueeze(0)  # (1, coils, H, W, 2)
    
            with torch.no_grad():
                output = self.model(kspace_slice.to(self.device), mask_torch).cpu() # (1, H, W)

            output = CenterCrop(crop_size)(output)
            output = abs(output - output.mean())
            output = CenterCrop(orig_size)(output)
    
            outputs.append(output)
        recon = torch.cat(outputs, dim=0).unsqueeze(1)  # (slices, 1, H, W)

        return recon
