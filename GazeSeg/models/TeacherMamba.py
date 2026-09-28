import torch
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import functional as F
from .GazeEncoder import GazeEncoder
from .VisionEncoder import ViGBackbone, Upsample, GNNBlock
from .NdMamba import RMSNorm

def sample_img_features(img_feat, traj):
    B, C, H, W = img_feat.shape
    B2, n, _ = traj.shape
    grid = traj * 2 - 1
    grid = grid.unsqueeze(2)  # (B, n, 1, 2)
    img_feat_exp = img_feat.unsqueeze(1).repeat(1, n, 1, 1, 1)
    out = F.grid_sample(
        img_feat_exp.view(B * n, C, H, W),
        grid.view(B * n, 1, 1, 2),
        mode="bilinear",
        align_corners=True
    )
    return out.view(B, n, C)

class Fusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.gaze_q_proj = nn.Linear(dim, dim)
        self.img_k_proj = nn.Linear(dim, dim)
        self.gaze_out_proj = nn.Linear(dim, dim)

        self.img_norm = nn.LayerNorm(dim)
        self.gaze_norm = nn.LayerNorm(dim)

        self.local_heads = num_heads // 2
        self.global_heads = num_heads - self.local_heads

    def forward(self, img_feature, gaze_feature):
        B, D, H, W = img_feature.shape
        L = gaze_feature.shape[1]

        img_seq = img_feature.flatten(2).transpose(1, 2)
        N = img_seq.shape[1]

        img_cross_out, _ = self.cross_attn(
            query=img_seq,
            key=gaze_feature,
            value=gaze_feature
        )
        img1_seq = self.img_norm(img_seq + img_cross_out)

        gaze_q = self.gaze_q_proj(gaze_feature)  # [B, L, D]
        img_k = self.img_k_proj(img1_seq)  # [B, N, D]
        gaze_q = F.normalize(gaze_q, dim=-1)
        img_k = F.normalize(img_k, dim=-1)

        gaze_q = gaze_q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        img_k = img_k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(gaze_q, img_k.transpose(-2, -1)) * scale

        local_scores = attn_scores[:, :self.local_heads, :, :].max(dim=-1)[0]  # [B, local_heads, L]
        global_scores = attn_scores[:, self.local_heads:, :, :].mean(dim=-1)  # [B, global_heads, L]
        all_head_scores = torch.cat([local_scores, global_scores], dim=1)

        avg_scores = all_head_scores.mean(dim=1)

        weights = torch.sigmoid(avg_scores)  # [B, L]
        weights = weights.unsqueeze(-1)

        gaze_out = weights * gaze_feature
        gaze_out = self.gaze_norm(self.gaze_out_proj(gaze_out) + gaze_feature)

        img_out = img1_seq.transpose(1, 2).reshape(B, D, H, W)

        return img_out, gaze_out

class TeacherMamba(torch.nn.Module):
    def __init__(self):
        super(TeacherMamba, self).__init__()
        self.VisionEncoder = ViGBackbone()
        save_model = torch.load('models/pretrained_models/pvig_s_82.1.pth.tar', map_location='cpu')
        model_dict = self.VisionEncoder.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.VisionEncoder.load_state_dict(model_dict)

        self.GazeEncoder = GazeEncoder()

        self.phy_proj = nn.Linear(7, 80)
        self.phy_norm = RMSNorm(80)
        self.gate_proj = nn.Linear(80 * 2, 80)

        self.fusion_dims = [80, 160, 400, 640]
        self.fusion_blocks = nn.ModuleList([
            Fusion(dim=dim, num_heads=4) for dim in self.fusion_dims
        ])

        self.upsample_1 = Upsample(self.fusion_dims[3], self.fusion_dims[3], 2, 2)
        self.up_residual_conv1 = GNNBlock(self.fusion_dims[3] + self.fusion_dims[2], self.fusion_dims[2])
        self.upsample_2 = Upsample(self.fusion_dims[2], self.fusion_dims[2], 2, 2)
        self.up_residual_conv2 = GNNBlock(self.fusion_dims[2] + self.fusion_dims[1], self.fusion_dims[1])
        self.upsample_3 = Upsample(self.fusion_dims[1], self.fusion_dims[1], 2, 2)
        self.up_residual_conv3 = GNNBlock(self.fusion_dims[1] + self.fusion_dims[0], self.fusion_dims[0])
        self.segmentation_head = nn.Sequential(
            Conv2d(self.fusion_dims[0], self.fusion_dims[0], 3, 1, 1), nn.BatchNorm2d(self.fusion_dims[0]), nn.ReLU(),
            Upsample(self.fusion_dims[0], self.fusion_dims[0] // 2, 2, 2), nn.BatchNorm2d(self.fusion_dims[0] // 2), nn.ReLU(),
            Conv2d(self.fusion_dims[0] // 2, self.fusion_dims[0] // 2, 3, 1, 1), nn.BatchNorm2d(self.fusion_dims[0] // 2), nn.ReLU(),
            Upsample(self.fusion_dims[0] // 2, self.fusion_dims[0] // 4, 2, 2), nn.BatchNorm2d(self.fusion_dims[0] // 4), nn.ReLU(),
            Conv2d(self.fusion_dims[0] // 4, self.fusion_dims[0] // 4, 3, 1, 1), nn.BatchNorm2d(self.fusion_dims[0] // 4), nn.ReLU(),
            Conv2d(self.fusion_dims[0] // 4, 1, 1, 1, bias=True)
        )

        self.deep_sup_d1 = nn.Conv2d(self.fusion_dims[2], 3, 1, 1, 0)
        self.deep_sup_d2 = nn.Conv2d(self.fusion_dims[1], 3, 1, 1, 0)
        self.deep_sup_d3 = nn.Conv2d(self.fusion_dims[0], 3, 1, 1, 0)

    def gaze_stem(self, gaze_sem_feature, gaze_phy_feature):
        phy_proj = self.phy_proj(gaze_phy_feature)
        phy_proj = self.phy_norm(phy_proj)

        concat_feat = torch.cat([gaze_sem_feature, phy_proj], dim=-1)
        gate = torch.sigmoid(self.gate_proj(concat_feat))  # 门控值范围0~1

        fused_feat = gate * gaze_sem_feature + (1 - gate) * phy_proj
        return fused_feat

    def forward(self, image, traj):
        if image.size(1) == 1:
            image = image.repeat(1, 3, 1, 1)
        img_feature = self.VisionEncoder.stem(image) + self.VisionEncoder.pos_embed
        gaze_sem_feature = sample_img_features(img_feature, traj[:, :, :2])
        gaze_phy_feature = traj
        gaze_feature = self.gaze_stem(gaze_sem_feature, gaze_phy_feature)

        j = 0
        img_feature_list = []
        for i in range(len(self.VisionEncoder.backbone)):
            img_feature = self.VisionEncoder.backbone[i](img_feature)
            gaze_feature = self.GazeEncoder.backbone[i](gaze_feature)
            if i in [1, 4, 11, 14]:
                img_feature, gaze_feature = self.fusion_blocks[j](img_feature, gaze_feature)
                img_feature_list.append(img_feature)
                j += 1

        f1, f2, f3, f4 = img_feature_list[0], img_feature_list[1], img_feature_list[2], img_feature_list[3]
        d1 = self.upsample_1(f4)  # 640x14x14
        d1 = self.up_residual_conv1(torch.cat([d1, f3], dim=1))  # 400x14x14 # 1040x14x14

        d2 = self.upsample_2(d1)  # 400x28x28
        d2 = self.up_residual_conv2(torch.cat([d2, f2], dim=1))  # 160x28x28 # 560x28x28

        d3 = self.upsample_3(d2)  # 160x56x56
        d3 = self.up_residual_conv3(torch.cat([d3, f1], dim=1))  # 80x56x56 # 240x56x56
        seg_logits = self.segmentation_head(d3)  # 1x224x224

        output_d1 = self.deep_sup_d1(d1)
        output_d2 = self.deep_sup_d2(d2)
        output_d3 = self.deep_sup_d3(d3)

        return {'logits': seg_logits, 'features': [f1, f2, f3, f4], 'deep_sup': [output_d1, output_d2, output_d3]}
