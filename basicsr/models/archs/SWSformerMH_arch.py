import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base


class SWSAttention(nn.Module):
    def __init__(self, c, num_heads, win_size, sub_win_size, qkv_bias, qk_scale, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.c = c
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.win_size = win_size
        self.sub_win_size = sub_win_size
        self.num_heads = num_heads
        self.q = nn.Linear(c, c, bias=qkv_bias)
        self.kv = nn.Linear(c, c * 2, bias=qkv_bias)
        self.scale = qk_scale or (c // num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(c, c)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, attn=None):
        B, C, H, W = x.shape
        x, win_size, sub_win_size, nWh = self.check_image_size(x)

        win_x = rearrange(x, 'B C (nWh wH) (nWw wW) -> B (nWh nWw) wH wW C', wH=win_size[0], wW=win_size[1])
        sub_win_x = rearrange(win_x, 'B nW (nWh wH) (nWw wW) C -> B nW (nWh nWw) (wH wW) C', wH=sub_win_size[0], wW=sub_win_size[1])
        shuffled_sub_win_x = sub_win_x.transpose(1, 2)

        pooled_shuffled_sub_win_x = shuffled_sub_win_x.mean(-2)
        merged_shuffled_sub_win_x = rearrange(shuffled_sub_win_x, 'B sub_nW nW N C -> B sub_nW (nW N) C')

        q = self.q(pooled_shuffled_sub_win_x)
        k, v = self.kv(merged_shuffled_sub_win_x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'B sub_nW N (nH d) -> B sub_nW nH N d', nH=self.num_heads), [q, k, v])
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        shuffled_attn_x = attn @ v

        attn_x = rearrange(shuffled_attn_x, 'B sub_nW nH N d -> B N sub_nW (nH d)')
        attn_x = self.proj(attn_x)
        attn_x = self.proj_drop(attn_x)

        out = attn_x.unsqueeze(-2).repeat(1, 1, 1, sub_win_size[0] * sub_win_size[1], 1)
        out = rearrange(out, 'B nW (nWh nWw) (wH wW) C -> B nW (nWh wH) (nWw wW) C', nWh=win_size[0] // sub_win_size[0], wH=sub_win_size[0])
        out = rearrange(out, 'B (nWh nWw) wH wW C -> B C (nWh wH) (nWw wW)', nWh=nWh, wH=win_size[0])
        
        return out[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.shape

        win_h = max(h // self.win_size, 1)
        win_w = max(w // self.win_size, 1)

        sub_win_h = min(self.sub_win_size, win_h)
        sub_win_w = min(self.sub_win_size, win_w)

        win_h += (sub_win_h - win_h % sub_win_h) % sub_win_h
        win_w += (sub_win_w - win_w % sub_win_w) % sub_win_w

        mod_pad_h = (win_h - h % win_h) % win_h
        mod_pad_w = (win_w - w % win_w) % win_w
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

        nWh = x.shape[2] // win_h
        win_size = (win_h, win_w)
        sub_win_size = (sub_win_h, sub_win_w)
        self.win_size_for_flops = [win_size, sub_win_size]
        return x, win_size, sub_win_size, nWh


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class PEG(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.pos = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, padding=k // 2, stride=1, groups=c, bias=True)

    def forward(self, x):
        x = self.pos(x) + x
        return x


class SWSBlock(nn.Module):
    def __init__(self, c, num_heads, win_size, sub_win_size, nwc_k, qkv_bias, qk_scale, attn_drop=0., proj_drop=0., FFN_Expand=2.66, drop_out_rate=0.):
        super().__init__()
        
        self.pos_block = PEG(c)
        self.attn = SWSAttention(c, num_heads, win_size, sub_win_size, qkv_bias, qk_scale)
        self.nwc = nn.Conv2d(c, c, kernel_size=nwc_k, stride=1, padding=nwc_k//2, groups=c, bias=True)
        self.gdfn = FeedForward(c, FFN_Expand, bias=False)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.norm3 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.attn(self.pos_block(x))
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.nwc(x)
        y = y + x * self.alpha

        x = self.norm3(y)
        x = self.gdfn(x)
        x = self.dropout2(x)
        y = y + x * self.gamma
        
        return y

class SWSformerMH(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], train_size=256,
        win_size=16, sub_win_size=8, nwc_ks=[7, 5, 3, 3, 3], num_heads=[1, 2, 4, 8, 16], qkv_bias=True, qk_scale=None, n_heads=4, combinate_heads=False):
        super().__init__()
        
        self.n_heads = n_heads
        self.combinate_heads = combinate_heads

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel*n_heads, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
    
        chan = width
        for nB, nwc_k, nH in zip(enc_blk_nums, nwc_ks, num_heads):
            self.encoders.append(
                nn.Sequential(
                    *[SWSBlock(chan, nH, win_size, sub_win_size, nwc_k, qkv_bias, qk_scale, train_size) for _ in range(nB)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
            train_size = train_size // 2

        self.middle_blks = \
            nn.Sequential(
                *[SWSBlock(chan, num_heads[-1], win_size, sub_win_size, nwc_ks[-1], qkv_bias, qk_scale, train_size) for _ in range(middle_blk_num)]
            )

        for nB, nwc_k, nH in zip(dec_blk_nums, nwc_ks[-2::-1], num_heads[-2::-1]):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            train_size = train_size * 2
            self.decoders.append(
                nn.Sequential(
                    *[SWSBlock(chan, nH, win_size, sub_win_size, nwc_k, qkv_bias, qk_scale, train_size) for _ in range(nB)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
 
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        x = self.ending(x)
        x = x.view(B, self.n_heads, C, H, W)
        if self.combinate_heads:
            x_ = []
            for i in range(self.n_heads):
                for j in range(i+1):
                    x_.append((x[:, i] + x[:, j])/2)
            x = torch.stack(x_, dim=1)
        x = x + inp.unsqueeze(1)

        return x[:, :, :, :H, :W] # Batch, N_heads, C, H, W

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class LocalSWSAttention(SWSAttention):
    def __init__(self, c, num_heads, win_size, sub_win_size, qkv_bias, qk_scale, base_size=None, kernel_size=None, train_size=None):
        super().__init__(c, num_heads, win_size, sub_win_size, qkv_bias, qk_scale)
        self.base_size = base_size
        self.kernel_size = kernel_size
        self.train_size = train_size

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def _forward(self, x):
        B, C, H, W = x.shape
        x, win_size, sub_win_size, nWh = self.check_image_size(x)

        win_x = rearrange(x, 'B C (nWh wH) (nWw wW) -> B (nWh nWw) wH wW C', wH=win_size[0], wW=win_size[1])
        sub_win_x = rearrange(win_x, 'B nW (nWh wH) (nWw wW) C -> B nW (nWh nWw) (wH wW) C', wH=sub_win_size[0], wW=sub_win_size[1])
        shuffled_sub_win_x = sub_win_x.transpose(1, 2)

        pooled_shuffled_sub_win_x = shuffled_sub_win_x.mean(-2)
        merged_shuffled_sub_win_x = rearrange(shuffled_sub_win_x, 'B sub_nW nW N C -> B sub_nW (nW N) C')

        q = self.q(pooled_shuffled_sub_win_x)
        k, v = self.kv(merged_shuffled_sub_win_x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'B sub_nW N (nH d) -> B sub_nW nH N d', nH=self.num_heads), [q, k, v])
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        shuffled_attn_x = attn @ v

        attn_x = rearrange(shuffled_attn_x, 'B sub_nW nH N d -> B N sub_nW (nH d)')
        attn_x = self.proj(attn_x)
        attn_x = self.proj_drop(attn_x)

        out = attn_x.unsqueeze(-2).repeat(1, 1, 1, sub_win_size[0] * sub_win_size[1], 1)
        out = rearrange(out, 'B nW (nWh nWw) (wH wW) C -> B nW (nWh wH) (nWw wW) C', nWh=win_size[0] // sub_win_size[0], wH=sub_win_size[0])
        out = rearrange(out, 'B (nWh nWw) wH wW C -> B C (nWh wH) (nWw wW)', nWh=nWh, wH=win_size[0])

        return out[:, :, :H, :W]

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

        x = self.grids(x) # convert to local windows 
        out = self._forward(x)
        out = self.grids_inverse(out) # reverse
        return out


def replace_layers(model, base_size, train_size, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, **kwargs)

        if isinstance(m, SWSAttention):
            attn = LocalSWSAttention(m.c, m.num_heads, m.win_size, m.sub_win_size, m.qkv_bias, m.qk_scale, base_size=base_size, train_size=train_size)
            setattr(model, n, attn)

class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)

class SWSformerMHLocal(Local_Base, SWSformerMH):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        SWSformerMH.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp, **kwargs)