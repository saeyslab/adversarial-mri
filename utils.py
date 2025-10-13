import numpy as np

import pywt

import requests

import cv2 as cv

import sigpy

from tqdm import tqdm

import torch

from typing import Tuple, Optional

import os, csv

from pathlib import Path

import matplotlib.pyplot as plt

import torch.nn.functional as F

import csv

def _out_root():
    base = os.environ.get("VSC_SCRATCH")
    r = Path(base) / "fastMRI" if base else Path("./outputs/fastMRI")
    (r / "fig6").mkdir(parents=True, exist_ok=True)
    (r / "csv").mkdir(parents=True, exist_ok=True)
    return r

def _save_fig6(path, y0, y_tgt, y_adv, x, x_adv):     # for diff -- (y_adv-yo)

    def to2d(t):                # for both it takes H,W
        t = t.detach().float().cpu()
        if t.dim() == 4:        # [B,1,H,W]
            t = t[0, 0]
        elif t.dim() == 3:      # [1,H,W]
            t = t[0]  
        return t.squeeze()

    x0   = to2d(x)
    xad  = to2d(x_adv)
    y0_  = to2d(y0)
    ytg  = to2d(y_tgt)
    yad  = to2d(y_adv)

    a = y0_.numpy().ravel()
    vmin = float(np.percentile(a, 1))
    vmax = float(np.percentile(a, 99))
    if vmax <= vmin:
        vmin, vmax = float(a.min()), float(a.max())

    diff = (yad - y0_).abs()
    dmax = float(diff.max().item())
    diff_n = diff / dmax if dmax > 0 else diff

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6), dpi=120)

    top_imgs   = [x0, xad, diff_n]
    top_titles = ["x", "x_adv", "|y_adv - y0| (norm)"]
    for j in range(3):
        ax = axes[0, j]
        if j < 2:  
            ax.imshow(top_imgs[j], cmap="gray", vmin=vmin, vmax=vmax)
        else:      
            ax.imshow(top_imgs[j], cmap="gray")
        ax.set_title(top_titles[j], fontsize=9)
        ax.axis("off")

    bottom_imgs   = [y0_, ytg, yad]
    bottom_titles = ["y0", "y_tgt = y0 + xdet", "y_adv"]
    for j in range(3):
        ax = axes[1, j]
        ax.imshow(bottom_imgs[j], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(bottom_titles[j], fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)





def get_hw(like_tensor: torch.Tensor) -> Tuple[int, int]:
    assert like_tensor.dim() == 4, "like_tensor must be (B,C,H,W)"
    return int(like_tensor.shape[-2]), int(like_tensor.shape[-1])

def draw_mask_hw(
    H: int, W: int,
    kind: str = "square",
    size: int = 20,
    thickness: int = -1,
    value: float = 1.0,
    center: Optional[Tuple[int,int]] = None
) -> np.ndarray:
    img = np.zeros((H, W), dtype=np.float32)
    cy, cx = (H // 2, W // 2) if center is None else (int(center[0]), int(center[1]))

    if kind == "square":
        half = size // 2    
        y1, y2 = max(0, cy - half), min(H - 1, cy + half)    
        x1, x2 = max(0, cx - half), min(W - 1, cx + half)
        cv.rectangle(img, (x1, y1), (x2, y2), color=float(value), thickness=int(thickness))

    elif kind == "line":  
        cy = max(0, min(H - 1, cy))
        cx = max(0, min(W - 1, cx))
        
        half_len = max(1, int(size) // 2)                            # size = line length         
        x1 = max(0, cx - half_len)
        x2 = min(W - 1, cx + half_len)

        if x1 == x2:
            x2 = min(W - 1, x1 + 1)
    
        cv.line(
            img,
            (x1, cy), (x2, cy),
            color=float(value),
            thickness=int(max(1, thickness if thickness != -1 else 5)),
        )


    elif kind == "circle":
        pass


    else:
        raise ValueError(f"Invalid kind: {kind}")

    return img

def cv_mask_to_torch_like(mask_hw: np.ndarray, like_tensor: torch.Tensor) -> torch.Tensor:
    assert like_tensor.dim() == 4, "like_tensor must be (B,C,H,W)"
    B, C, H, W = like_tensor.shape
    assert mask_hw.shape == (H, W), f"mask and target not matching: {mask_hw.shape} vs {(H,W)}"
    m = torch.from_numpy(mask_hw).to(like_tensor.device, dtype=like_tensor.dtype)
    m = m.view(1, 1, H, W).expand(B, C, H, W).contiguous()
    return m

def make_xdet_cv_like(
    like_tensor: torch.Tensor,
    kind: str = "square",
    size: int = 20,
    thickness: int = -1,
    value: float = 1.0,
    center: Optional[Tuple[int,int]] = None
) -> torch.Tensor:
    H, W = get_hw(like_tensor)
    mask_hw = draw_mask_hw(H, W, kind=kind, size=size, thickness=thickness, value=value, center=center)
    return cv_mask_to_torch_like(mask_hw, like_tensor)


## filing resp. mask - organ - model 
def make_output_dirs_by_mask_first(base_dir, mask_type, organ, model_tag):
    base = Path(base_dir)
    csv_dir = base / "csv" / mask_type / organ
    fig6_dir = base / "fig6" / mask_type / organ / model_tag
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig6_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, fig6_dir


def append_csv_row(csv_dir, model_tag, row_dict):
    csv_path = csv_dir / f"{model_tag}.csv"
    write_header = (not csv_path.exists())
    with open(csv_path, "a", newline = "") as f:
        w = csv.DictWriter(f, fieldnames = list(row_dict.keys()))
        if write_header: w.writeheader()
        w.writerow(row_dict)
        
        

def overlay_mask(gray_image, x_hall, alpha=.5, q=.25):
    image_bgr = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)

    red_mask = np.zeros_like(image_bgr)
    red_mask[:, :, 0] = 1
    mask = (x_hall >= q)

    image_bgr[mask] = cv.addWeighted(image_bgr[mask], 1 - alpha, red_mask[mask], alpha, 0)
    
    return normalize(image_bgr)

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def image_to_kspace(image):
    if isinstance(image, np.ndarray):
        kspace = np.fft.fft2(np.fft.fftshift(image, axes=[-1, -2]), axes=[-1, -2])
    elif isinstance(image, torch.Tensor):
        kspace = torch.fft.fft2(torch.fft.fftshift(image, dim=(-2, -1)), dim=(-2, -1))
    else:
        raise TypeError("Invalid data type")
    return kspace

def kspace_to_image(ksp):
    if isinstance(ksp, np.ndarray):
        image = np.fft.ifftshift(np.fft.ifft2(ksp, axes=[-1, -2]), axes=[-1, -2])
    elif isinstance(ksp, torch.Tensor):
        image = torch.fft.ifftshift(torch.fft.ifft2(ksp, dim=(-2, -1)), dim=(-2, -1))
    else:
        raise TypeError("Invalid data type")
    return image.real

def download_model(url: str, fname: str):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 1 * 1024 * 1024  # 1 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)

    progress_bar.close()

def sparsity_norm(x, wavelet='coif5', level=2):
    coeffs = pywt.wavedec2(x, wavelet=wavelet, level=level)
    detail_coeffs = coeffs[1:]

    l1_norm = sum(np.sum(np.abs(detail)) for level in detail_coeffs for detail in level)

    return l1_norm

def zero_fill(sample: 'Sample') -> np.ndarray:
    return rss(sigpy.ifft(sample.masked_kspace, axes=(-1, -2)))

def rss(img):
    if isinstance(img, np.ndarray):
        return np.sqrt(np.sum(np.square(abs(img)), axis=1, keepdims=True))
    elif isinstance(img, torch.Tensor):
        return torch.sqrt(torch.sum(torch.square(abs(img)), dim=1, keepdims=True))
    else:
        raise TypeError("Invalid data type")
