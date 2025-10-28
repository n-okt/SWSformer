import torch
import math
import numpy as np
import seaborn as sns
from PIL import Image
from os import path as osp
from einops import rearrange
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.colors import Normalize
from basicsr.utils import imwrite, tensor2img
from basicsr.models.archs.SWSformer_arch import SWSAttention


class VisualizeAttnMap:
    def __init__(self, model, query_coord):
        self.attn_map = None
        self.input_size = None
        self.win_sizes = None
        self.query_coord = query_coord

        target_module = None
        for name, module in model.named_modules():
            if name.startswith('module.decoders') and isinstance(module, SWSAttention):
                target_module = module

        if target_module is not None:
            hook_handle = target_module.register_forward_hook(self.attn_map_hook)
        else:
            raise ValueError("SWSAttention module was not found.")

    def attn_map_hook(self, module, input, output):
        self.attn_map = module.attn_map
        self.input_size = input[0].shape[2:]
        self.win_sizes = module.win_size_for_flops

    def visualize(self, save_img_dir, alpha, img_name, visuals):
        attn_map = self.attn_map

        B, num_shuffled_win, nH, nQ, nK = attn_map.shape
        H, W = self.input_size
        x, y = self.query_coord
        win_size = self.win_sizes[0]
        nW = (H // win_size[0], W // win_size[1])
        sub_win_size = self.win_sizes[1]
        sub_nW = (nW[0] // sub_win_size[0], nW[1] // sub_win_size[1])
        
        if B > 1:
            raise ValueError("Batch size must be 1.")
        if H < y or W < x:
            raise ValueError(f"Query coord must be smaller than input size. ({y}, {x}) < ({H}, {W})")

        win_coord = (math.ceil(y / win_size[0]), math.ceil(x / win_size[1]))
        sub_win_coord = (math.ceil(y / sub_win_size[0]), math.ceil(x / sub_win_size[1]))
        sub_win_coord = [sub_win_coord[i] % 2 for i in range(2)]
        sub_win_coord = [sub_nW[i] if sub_win_coord[i] == 0 else sub_win_coord[i] for i in range(2)]

        win_idx = ((win_coord[0] - 1) * win_size[1] + win_coord[1]) - 1
        sub_win_idx = ((sub_win_coord[0] - 1) * sub_nW[1] + sub_win_coord[1]) - 1
        
        attn_map = attn_map.mean(2)
        attn_map = attn_map[:, sub_win_idx, win_idx, :]

        attn_map = attn_map.unsqueeze(1).repeat(1, sub_nW[0] * sub_nW[1], 1)
        mask_idx = list(range(sub_nW[0] * sub_nW[1]))
        mask_idx.remove(sub_win_idx)
        attn_map[:, mask_idx, :] = 0
        attn_map = rearrange(attn_map, 'B sub_nW (W sub_W) -> B W sub_nW sub_W', W=win_size[0]*win_size[1])
        attn_map = rearrange(attn_map, 'B W (sub_nWh sub_nWw) (sub_wH sub_wW) -> B W (sub_nWh sub_wH) (sub_nWw sub_wW)', sub_nWh=sub_nW[0], sub_wH=sub_win_size[0])
        attn_map = rearrange(attn_map, 'B (nWh nWw) wH wW -> B (nWh wH) (nWw wW)', nWh=win_size[0])
        attn_map = attn_map.squeeze(0).detach().cpu().numpy()
        
        cmap = plt.get_cmap('Reds')
        normalizer = Normalize(vmin=attn_map.min(), vmax=attn_map.max())
        heatmap = cmap(normalizer(attn_map))
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = heatmap[:, :, :3]

        gt = tensor2img([visuals['gt']], rgb2bgr=False)
        lq = tensor2img([visuals['lq']], rgb2bgr=False)

        composite = (1 - alpha) * gt + alpha * heatmap
        composite = composite.astype(np.uint8)

        x1 = (x // sub_win_size[0]) * sub_win_size[0]
        x2 = x1 + sub_win_size[0]
        y1 = (y // sub_win_size[1]) * sub_win_size[1]
        y2 = y1 + sub_win_size[1]

        blue_color = (0, 0, 255)
        lq[y1:y2, x1:x2, :] = blue_color
        heatmap[y1:y2, x1:x2, :] = blue_color
        composite[y1:y2, x1:x2, :] = blue_color

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, img in enumerate([lq, heatmap, composite]):
            ax = axes[i]
            ax.imshow(img)
            ax.axis('off')

        save_img_path = osp.join(save_img_dir, f'{img_name}.png')
        plt.savefig(save_img_path)
        plt.clf()
        plt.close()