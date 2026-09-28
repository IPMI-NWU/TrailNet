import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Conv2d
from github_version.MambaGazeSeg.models.VisionEncoder import ViGBackbone, Upsample, GNNBlock


class DistillLayer(nn.Module):
    def __init__(self, channels):

        super(DistillLayer, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.proj(x)

class StudentUNet(nn.Module):
    def __init__(self):
        super(StudentUNet, self).__init__()
        filters = [80, 160, 400, 640]

        self.VisionEncoder = ViGBackbone()
        save_model = torch.load('models/pretrained_models/pvig_s_82.1.pth.tar', map_location='cpu')
        model_dict = self.VisionEncoder.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.VisionEncoder.load_state_dict(model_dict)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = GNNBlock(filters[3] + filters[2], filters[2])
        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = GNNBlock(filters[2] + filters[1], filters[1])
        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = GNNBlock(filters[1] + filters[0], filters[0])
        self.segmentation_head = nn.Sequential(
            Conv2d(filters[0], filters[0], 3, 1, 1), nn.BatchNorm2d(filters[0]), nn.ReLU(),
            Upsample(filters[0], filters[0]//2, 2, 2), nn.BatchNorm2d(filters[0]//2), nn.ReLU(),
            Conv2d(filters[0]//2, filters[0]//2, 3, 1, 1), nn.BatchNorm2d(filters[0]//2), nn.ReLU(),
            Upsample(filters[0]//2, filters[0] // 4, 2, 2), nn.BatchNorm2d(filters[0]//4), nn.ReLU(),
            Conv2d(filters[0] // 4, filters[0] // 4, 3, 1, 1), nn.BatchNorm2d(filters[0]//4), nn.ReLU(),
            Conv2d(filters[0] // 4, 1, 1, 1, bias=True)
        )
        self.distill_layer_1 = DistillLayer(filters[0])
        self.distill_layer_2 = DistillLayer(filters[1])
        self.distill_layer_3 = DistillLayer(filters[2])
        self.distill_layer_4 = DistillLayer(filters[3])

        self.deep_sup_d1 = nn.Conv2d(filters[2], 3, 1, 1, 0)
        self.deep_sup_d2 = nn.Conv2d(filters[1], 3, 1, 1, 0)
        self.deep_sup_d3 = nn.Conv2d(filters[0], 3, 1, 1, 0)

    def forward(self, image):
        f1, f2, f3, f4 = self.VisionEncoder.forward(image)

        d1 = self.upsample_1(f4)  # 640x14x14
        d1 = self.up_residual_conv1(torch.cat([d1, f3], dim=1))  # 400x14x14 # 1040x14x14
        d2 = self.upsample_2(d1)  # 400x28x28
        d2 = self.up_residual_conv2(torch.cat([d2, f2], dim=1))  # 160x28x28 # 560x28x28
        d3 = self.upsample_3(d2)  # 160x56x56
        d3 = self.up_residual_conv3(torch.cat([d3, f1], dim=1))  # 80x56x56 # 240x56x56
        seg_logits = self.segmentation_head(d3)  # 1x224x224

        l1 = self.distill_layer_1(f1)
        l2 = self.distill_layer_2(f2)
        l3 = self.distill_layer_3(f3)
        l4 = self.distill_layer_4(f4)

        output_d1 = self.deep_sup_d1(d1)
        output_d2 = self.deep_sup_d2(d2)
        output_d3 = self.deep_sup_d3(d3)

        return {'logits': seg_logits, 'features': [l1, l2, l3, l4], 'deep_sup': [output_d1, output_d2, output_d3]}
