import torch.nn as nn
import torch
from .NdMamba import RMSNorm, NdMamba2_1d


class GazeEncoder(nn.Module):
    def __init__(self, blocks=None, channels=None, drop_path_rate=0.1):
        super().__init__()
        if blocks is None:
            blocks = [2, 2, 6, 2]
        if channels is None:
            channels = [80, 160, 400, 640]
        self.blocks = blocks
        self.channels = channels

        self.backbone = nn.ModuleList([])
        depth = sum(blocks)
        cur_depth = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Projector(channels[i-1], channels[i]))
            for j in range(blocks[i]):
                block_drop = drop_path_rate * cur_depth / max(depth - 1, 1)
                self.backbone.append(
                    GazeMambaBlock(
                        dim=channels[i],
                        drop_path=block_drop
                    )
                )
                cur_depth += 1

        self.final_norm = RMSNorm(channels[-1])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)

        x = self.final_norm(x)
        return x

class Projector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = RMSNorm(out_channels)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class GazeMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, ffn_expand=4, drop_path=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.mamba = NdMamba2_1d(
            cin = dim, cout = dim, cmid = dim // 80 * 64
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expand),
            nn.GELU(),
            nn.Linear(dim * ffn_expand, dim),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.mamba(x)
        x = residual + self.drop_path1(x)

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.drop_path2(x)
        return x
