import torch
import torch.nn as nn

import numpy as np

from models.model import Model

from tqdm import trange

from utils import normalize

class TargetedFGSM:
    def __init__(self, model: Model, eps=0.02, step_size=1e-5, n_iter=10, loss_func=None):
        self.model = model
        self.eps = float(eps)
        self.step_size = float(step_size)
        self.n_iter = int(n_iter)
        self.loss_func = loss_func if loss_func is not None else nn.MSELoss()

    def _to_4d_float_tensor(self, x, device, dtype=torch.float32):
        if isinstance(x, dict):  
            for k in ("image","img","zf_img","zf","x","input"):
                if k in x: x = x[k]; break
        elif hasattr(x, "image"): x = getattr(x, "image")
        elif hasattr(x, "img"):   x = getattr(x, "img")
        elif hasattr(x, "data"):  x = getattr(x, "data")

        if torch.is_tensor(x):
            t = x
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        elif isinstance(x, (list, tuple)):
            t = torch.as_tensor(np.array(x))
        else:
            raise TypeError(f"x_in type unsupported: {type(x)}")

        t = t.to(dtype)
        if t.dim() == 2:                # (H,W) --- (1,1,H,W)
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:
            if t.shape[0] in (1,3):     # (C,H,W) --- (1,C,H,W)
                t = t.unsqueeze(0)
            else:                        # (B,H,W) --- (B,1,H,W) - e.g batch:32 so this needs C -hannel
                t = t.unsqueeze(1)
        elif t.dim() != 4:
            raise ValueError(f"Expected 2D/3D/4D, got shape {tuple(t.shape)}")
        return t.to(device)

    def _to_mask_like(self, m, ref, device):
        if m is None: return None
        if torch.is_tensor(m): t = m
        elif isinstance(m, np.ndarray): t = torch.from_numpy(m)
        else: t = torch.as_tensor(m)
        if t.dim() == 2: t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3: t = t.unsqueeze(0)
        elif t.dim() != 4: raise ValueError(f"Mask dim must be 2/3/4, got {t.dim()}")
        return t.to(device=device, dtype=ref.dtype)

    def __call__(self, x_in, mask=None, alpha=0.3, w_in=1.0, w_out=1.0, patience=10):
        self.model.eval()
        device = next(self.model.parameters()).device

        x = self._to_4d_float_tensor(x_in, device)
        sigma, mu = torch.std_mean(x, dim=(-1, -2, -3), keepdim=True)
        sigma = sigma.clamp_min(1e-8)
        clip_min, clip_max = x.min(), x.max()

        with torch.no_grad():
            y0 = (self.model(x) - mu) / sigma

        m = self._to_mask_like(mask, y0, device) if mask is not None else None
        # y_tgt = y0 + alpha * m if m is not None else y0
        #if theres no mask, untargetted attaxk : y_tgt = y0
        if m is not None:
            y_rng = (y0.max() - y0.min()).detach()
            alpha_eff = alpha * y_rng if alpha <= 1.0 else torch.as_tensor(alpha, device=y0.device, dtype=y0.dtype)
            y_tgt = y0 + alpha_eff * m
        else:
            y_tgt = y0

        x_adv = torch.clamp(x.detach().clone() + self.step_size * (2*torch.rand_like(x) - 1), clip_min, clip_max)
        x_best = x_adv.clone()
        best_loss = np.inf
        timeout = 0

        progbar = trange(self.n_iter)
        for _ in progbar:
            x_adv.requires_grad_(True)
            y = (self.model(x_adv) - mu) / sigma

            if m is None:
                loss = self.loss_func(y, y0)
            else:
                loss1 = torch.square((y - y_tgt) * m).sum() / m.sum()
                loss2 = torch.square((y - y0) * (1 - m)).sum() / (1 - m).sum()
                loss = w_in * loss1 + w_out * loss2

            self.model.zero_grad(set_to_none=True)
            if x_adv.grad is not None: x_adv.grad.zero_()
            loss.backward()

            grad_sign = x_adv.grad.detach().sign()
            x_adv = x_adv.detach() - self.step_size * grad_sign

            delta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = (x + delta).clamp(clip_min, clip_max)

            if loss.item() < best_loss:
                best_loss = loss.item()
                x_best = x_adv.detach().clone()
                timeout = 0
            else:
                timeout += 1
            
            if timeout > patience:
                break

            progbar.set_postfix({'loss': loss.item(), 'best': best_loss})

        with torch.no_grad():
            y_adv = (self.model(x_best) - mu) / sigma

        return x_best, y_adv, y0, y_tgt, m
