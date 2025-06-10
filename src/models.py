# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.size()
        # squeeze
        y = x.mean((2,3))  # (B, C)
        y = F.relu(self.fc1(y), True)
        y = torch.sigmoid(self.fc2(y))  # (B, C)
        y = y.view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch)
        if in_ch != out_ch or stride != 1:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.down = nn.Identity()

    def forward(self, x):
        identity = self.down(x)
        out = F.relu(self.bn1(self.conv1(x)), True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + identity, True)

class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)   # 64×80×80
        self.layer2 = ResidualBlock(64, 128, stride=2)  # 128×40×40
        self.layer3 = ResidualBlock(128, 256, stride=2) # 256×20×20
        self.pool   = nn.AdaptiveAvgPool2d(1)           # 256×1×1
        self.fc     = nn.Linear(256, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(1)

def resnet18_ft():
    m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
    m.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
    m.fc    = nn.Linear(m.fc.in_features, 1)
    return m



MODEL_REGISTRY = {
    "small_cnn":  SmallResNet,
    "resnet18":   resnet18_ft,
}
