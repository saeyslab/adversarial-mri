import torch

import utils

from pathlib import Path

from fastmri.models import Unet as UN

from models.model import Model

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

    def _to_bchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:                                # [H,W]
            x = x.unsqueeze(0).unsqueeze(0)            # -- [1,1,H,W]
        elif x.ndim == 3:                              # [B,H,W] or [C,H,W]
            if x.shape[0] in (1, 3):                   # probably [C,H,W]
                x = x.unsqueeze(0)                     # -- [1,C,H,W]
                if x.shape[1] != 1:
                    x = x.mean(dim=1, keepdim=True) 
            else:                                   # [B,H,W]
                x = x.unsqueeze(1)                  # --- [B,1,H,W]
        elif x.ndim == 4:                           
            if x.shape[1] != 1:
                x = x.mean(dim=1, keepdim=True)     
        elif x.ndim == 5 and x.shape[2] == 1:       # [B,1,1,H,W] 
            x = x.squeeze(2)                        # -- [B,1,H,W]
        else:
            raise RuntimeError(f"Unsupported input shape: {x.shape}")
        return x

    def _preprocess(self, x: torch.Tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        mean = x.mean(dim=(-1, -2, -3), keepdim=True)
        std  = x.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-8)
        x = torch.clamp((x - mean) / std, -6, 6)
        return x, mean, std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(self.device, dtype=torch.float32)
        x = self._to_bchw(x)             
        x, mean, std = self._preprocess(x)
        y = self.model(x)
        y = y * std + mean                
        return y
