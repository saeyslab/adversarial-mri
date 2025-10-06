import torch

import numpy as np

import utils

from pathlib import Path

from fastmri.models import Unet as UN

from models.model import Model

from data import Sample

class UNet(Model):
    def __init__(self, subset: str, coil: str, weight_path: Path, config: dict, **kwargs):
        super().__init__(weight_path, **kwargs)

        self.model = UN(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0).to(self.device).eval()
        if not weight_path.exists():
            print('[*] Downloading weights...')

            baseurl = config['unet-url']
            url = config[f'unet-{subset}-{coil}']
            utils.download_model(f'{baseurl}/{url}', weight_path)
            print(f'[*] Weights saved to {weight_path}')
        else:
            print(f'[*] Loading weights from {weight_path}')
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model = torch.nn.DataParallel(self.model)

    def forward(self, sample: Sample, batch_size: int = 4) -> torch.tensor:
        images = torch.from_numpy(utils.zero_fill(sample)).float()
        mean, std = images.mean(dim=(-1, -2, -3), keepdim=True), images.std(dim=(-1, -2, -3), keepdim=True)
        images = torch.clamp((images - mean) / std, -6, 6)
    
        with torch.no_grad():
            num_batches = int(np.ceil(images.shape[0] / batch_size))
            outputs = []
            for b in range(num_batches):
                start, end = b*batch_size, min((b+1)*batch_size, images.shape[0])
                batch = images[start:end]
                output = self.model(batch.to(self.device)).cpu()

                outputs.append(output)
    
        results = torch.cat(outputs, dim=0)
        results = results * std + mean

        return results
