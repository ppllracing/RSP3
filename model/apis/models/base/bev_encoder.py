import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet18

from ...tools.config import Configuration

class BevEncoder(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        # 提取基本数据
        self.cfg = cfg
        # self.device = cfg.device
        self.dtype = cfg.dtype_model_torch
        self.model_params = cfg.bev_encoder_params

        trunk = resnet18(weights=None, zero_init_residual=True).to(self.dtype)
        
        self.conv1 = nn.Conv2d(
            self.model_params['inp_channel'], 
            self.model_params['mid_channel'], 
            kernel_size=7, stride=2, padding=3, bias=False, dtype=self.dtype
        )
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.max_pool = trunk.maxpool

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.layer4 = nn.Sequential(
            nn.Upsample(size=self.cfg.map_bev['map_down_sample'], mode='bilinear', align_corners=True),
            nn.Conv2d(self.model_params['mid_channel'] * 4, self.model_params['oup_channel'], kernel_size=5, padding=2, bias=False, dtype=self.dtype),
            nn.BatchNorm2d(self.model_params['oup_channel'], affine=True, dtype=self.dtype),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )

    def forward(self, x):
        # x = x.to(self.device)  # [B, C + 1, H, W]

        if x.shape[2] != 256 or x.shape[3] != 256:
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)  # [B, C, 256, 256]

        x = self.conv1(x)  # [B, C, 128, 128]
        x = self.bn1(x)  # [B, C, 128, 128]
        x = self.relu(x)  # [B, C, 128, 128]
        x = self.max_pool(x)  # [B, C, 64, 64]

        x = self.layer1(x)  # [B, C, 64, 64]
        x = self.layer2(x)  # [B, C * 2, 32, 32]
        x = self.layer3(x)  # [B, C * 4, 16, 16]
        x = self.layer4(x)  # [B, C * 4, *map_down_sample]
        x = torch.flatten(x, 2)  # [B, C * 4, map_down_sample[0] * map_down_sample[1]]
        x = x.permute(0, 2, 1)  # [B, map_down_sample[0] * map_down_sample[1], C * 4]
        return x
