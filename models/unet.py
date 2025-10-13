import torch

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

    def forward(self, sample: Sample) -> torch.tensor:
        if sample.is_numpy():
            images = torch.from_numpy(utils.rss(sample.image)).float().to(self.device)
        else:
            images = utils.rss(sample.image).float().to(self.device)
        mean, std = images.mean(dim=(-1, -2, -3), keepdim=True), images.std(dim=(-1, -2, -3), keepdim=True)
        images = torch.clamp((images - mean) / std, -6, 6)
    
        output = self.model(images) * std + mean

        return output
